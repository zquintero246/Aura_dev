"""Entrenamiento de Aura en TensorDock usando una sola GPU A100."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, Optional, Tuple

from array import array

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import numpy as np

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase, get_linear_schedule_with_warmup
except ImportError as exc:  # pragma: no cover - dependencia externa
    raise ImportError(
        "Se requiere transformers. Instala con `pip install transformers`."
    ) from exc

from AuraLLM.train_DDP.train_aura import (  # type: ignore
    Config as ModelConfig,
    CUSTOM_MODEL_PRESETS,
    GPT2,
    SpanishCorpus,
)


MODEL_PRESETS: Dict[str, Dict[str, int]] = {
    name: {
        "embed_size": preset["embed_size"],
        "num_layers": preset["num_layers"],
        "num_heads": preset["num_heads"],
        "seq_length": preset.get("seq_len", preset.get("seq_length", 1024)),
    }
    for name, preset in CUSTOM_MODEL_PRESETS.items()
}

MODEL_PRESETS.update(
    {
        "aura-72h-max": {
            "embed_size": 2048,
            "num_layers": 24,
            "num_heads": 16,
            "seq_length": 1024
        }
    }
)

HF_DATASET_PRESETS: Dict[str, Dict[str, object]] = {
    "oscar-es": {
        "hf_dataset_name": "oscar",
        "hf_dataset_config": "unshuffled_deduplicated_es",
        "hf_dataset_split": "train",
        "hf_text_field": "text",
        "hf_trust_remote_code": True,
    },
    "cc100-es": {
        "hf_dataset_name": "cc100",
        "hf_dataset_config": "es",
        "hf_dataset_split": "train",
        "hf_text_field": "text",
        "hf_trust_remote_code": True,
    },
    "mc4-es": {
        "hf_dataset_name": "mc4",
        "hf_dataset_config": "es",
        "hf_dataset_split": "train",
        "hf_text_field": "text",
        "hf_trust_remote_code": True,
    },
    "wikimedia-es": {
        "hf_dataset_name": "wikimedia/wikipedia",
        "hf_dataset_config": "20231101.es",
        "hf_dataset_split": "train",
        "hf_text_field": "text",
        "hf_trust_remote_code": True,
    },
    "wikicorpus-es": {
        "hf_dataset_name": "PlanTL-GOB-ES/wikicorpus-es",
        "hf_dataset_config": "2023-06-21",
        "hf_dataset_split": "train",
        "hf_text_field": "text",
        "hf_trust_remote_code": True,
    },
    "spanish-ewt": {
        "hf_dataset_name": "universal_dependencies",
        "hf_dataset_config": "es_ancora-ud-2.12",
        "hf_dataset_split": "train",
        "hf_text_field": "text",
        "hf_trust_remote_code": True,
    },
}


TOKENIZE_CHUNK_SIZE = 10_000
TOKEN_CACHE_FILENAME = "dataset_tokens.pt"
_TOKEN_CACHE_PATH: Optional[Path] = None
NAN_MIN_LR_DEFAULT = 1e-7
NAN_PATIENCE_DEFAULT = 5
GIB = 1024 ** 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento desde cero de Aura en TensorDock")
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=Path("AuraLLM/datasets/spanish_corpus"),
        help="Ruta al corpus o directorio con .txt/.jsonl (por defecto /datasets/spanish_corpus)",
    )
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        default="oscar",
        help="Dataset de Hugging Face a descargar automáticamente si no existe dataset_path",
    )
    parser.add_argument(
        "--hf_dataset_config",
        type=str,
        default="unshuffled_deduplicated_es",
        help="Configuración/subconjunto del dataset de Hugging Face",
    )
    parser.add_argument(
        "--hf_dataset_split",
        type=str,
        default="train",
        help="Split del dataset de Hugging Face",
    )
    parser.add_argument(
        "--hf_text_field",
        type=str,
        default="text",
        help="Campo que contiene el texto dentro del dataset de Hugging Face",
    )
    parser.add_argument(
        "--hf_download_limit",
        type=int,
        default=None,
        help="Limita el número de ejemplos descargados (None descarga todo el split)",
    )
    parser.add_argument(
        "--hf_streaming",
        action="store_true",
        help="Usa streaming de datasets para escribir directamente a disco",
    )
    parser.add_argument(
        "--hf_trust_remote_code",
        action="store_true",
        help="Permite ejecutar código remoto del dataset de Hugging Face (requerido para algunos datasets)",
    )
    parser.add_argument(
        "--hf_auth_token",
        type=str,
        default=None,
        help="Token de autenticación para datasets privados de Hugging Face",
    )
    parser.add_argument(
        "--hf_dataset_preset",
        type=str,
        choices=sorted(HF_DATASET_PRESETS.keys()),
        help="Atajo para rellenar los parámetros del dataset de Hugging Face",
    )
    parser.add_argument(
        "--skip_auto_download",
        action="store_true",
        help="Desactiva la descarga automática del dataset si no existe dataset_path",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directorio donde guardar checkpoints y modelos",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="auto",
        help="Nombre/ruta del tokenizer o 'auto' para entrenarlo desde el corpus",
    )
    parser.add_argument(
        "--tokenizer_vocab_size",
        type=int,
        default=52000,
        help="Tamaño del vocabulario si se entrena tokenizer automáticamente",
    )
    parser.add_argument(
        "--tokenizer_min_frequency",
        type=int,
        default=2,
        help="Frecuencia mínima de tokens para el tokenizer automático",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Número de épocas completas")
    parser.add_argument(
        "--model_preset",
        type=str,
        default="auto",
        choices=["auto", "manual", *sorted(MODEL_PRESETS.keys())],
        help="Preset de arquitectura (auto detecta según GPU)",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Tamaño de batch real (antes de acumulación)")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Pasos a acumular antes de optimizar",
    )
    parser.add_argument("--learning_rate", type=float, default=None, help="Tasa de aprendizaje inicial")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay para AdamW")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Pasos de warmup para el scheduler lineal")
    parser.add_argument("--seq_length", type=int, default=None, help="Longitud de contexto (tokens)")
    parser.add_argument("--embed_size", type=int, default=None, help="Dimensión de embeddings")
    parser.add_argument("--num_layers", type=int, default=None, help="Número de capas Transformer")
    parser.add_argument("--num_heads", type=int, default=None, help="Cabezas de atención por capa")
    parser.add_argument("--dropout", type=float, default=None, help="Probabilidad de dropout")
    parser.add_argument("--validation_split", type=float, default=0.01, help="Proporción para validación (0 desactiva)")
    parser.add_argument("--save_steps", type=int, default=None, help="Guardar checkpoint cada N pasos de optimización")
    parser.add_argument("--resume_from", type=Path, default=None, help="Ruta a un checkpoint previo")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    parser.add_argument("--num_workers", type=int, default=4, help="Workers de DataLoader")
    parser.add_argument(
        "--auto_num_workers",
        action="store_true",
        help="Ajusta num_workers automáticamente según los núcleos disponibles",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=None,
        help="Prefetch factor para DataLoader (requiere workers > 0)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Actualizar barra de progreso cada N pasos acumulados",
    )
    parser.add_argument(
        "--limit_train_tokens",
        type=int,
        default=None,
        help="Limita los tokens de entrenamiento tras el split (múltiplo de seq_length)",
    )
    parser.add_argument(
        "--limit_val_tokens",
        type=int,
        default=None,
        help="Limita los tokens de validación tras el split (múltiplo de seq_length)",
    )
    parser.add_argument(
        "--target_hours",
        type=float,
        default=36.0,
        help="Objetivo de duración total del entrenamiento en horas para optimizar el runtime",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=None,
        help="Activa checkpointing para capas Transformer (reduce VRAM, aumenta cómputo)",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
        help="Desactiva el checkpointing incluso cuando el preset lo activa automáticamente",
    )
    parser.add_argument(
        "--checkpoint_segments",
        type=int,
        default=4,
        help="Segmentos para checkpoint_sequential cuando está activo el gradient checkpointing",
    )
    parser.add_argument(
        "--nan_behavior",
        type=str,
        choices=("stop", "skip", "reduce-lr"),
        default="stop",
        help=(
            "Cómo actuar si la pérdida es NaN/Inf: 'stop' detiene el entrenamiento, "
            "'skip' ignora el microbatch y continúa, 'reduce-lr' además reduce la tasa"
            " de aprendizaje."
        ),
    )
    parser.add_argument(
        "--nan_lr_factor",
        type=float,
        default=0.5,
        help="Factor multiplicador para la LR cuando --nan_behavior=reduce-lr",
    )
    parser.add_argument(
        "--nan_min_lr",
        type=float,
        default=NAN_MIN_LR_DEFAULT,
        help=(
            "Límite inferior para la LR cuando --nan_behavior=reduce-lr."
            " Evita que la tasa de aprendizaje llegue a cero."
        ),
    )
    parser.add_argument(
        "--nan_patience",
        type=int,
        default=NAN_PATIENCE_DEFAULT,
        help=(
            "Número máximo de batches consecutivos con NaN permitidos antes de detener"
            " el entrenamiento. Solo aplica cuando la acción no es 'stop'."
        ),
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
        help=(
            "Precisión de entrenamiento. 'auto' usa bf16 cuando esté disponible, "
            "fp16 en otras GPUs y fp32 en CPU."
        ),
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Compila el modelo con torch.compile para aumentar el throughput",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=("default", "reduce-overhead", "max-autotune"),
        help="Modo de optimización para torch.compile",
    )
    parser.add_argument(
        "--compile_fullgraph",
        action="store_true",
        help="Fuerza torch.compile(fullgraph=True) (puede tardar más en compilar)",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Norma máxima para clipping de gradiente (None desactiva)",
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        dest="grad_clip",
        help="Alias de --grad_clip para compatibilidad",
    )

    args = parser.parse_args()

    if args.hf_dataset_preset:
        preset = HF_DATASET_PRESETS[args.hf_dataset_preset]
        for key, value in preset.items():
            setattr(args, key, value)

    args.output_dir = args.output_dir.expanduser().resolve()

    if args.auto_num_workers:
        cpu_count = os.cpu_count() or 1
        if cpu_count <= 2:
            suggested_workers = 0
        else:
            suggested_workers = max(2, min(8, cpu_count // 2))
        if suggested_workers != args.num_workers:
            print(
                "Ajuste automático de num_workers: "
                f"{args.num_workers} -> {suggested_workers}",
                flush=True,
            )
            args.num_workers = suggested_workers

    return args


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_model_preset(args: argparse.Namespace, total_mem_gb: float) -> None:
    preset_name = args.model_preset
    if preset_name == "auto":
        if total_mem_gb >= 70:
            preset_name = "aura-72h-max"
        elif total_mem_gb >= 55:
            preset_name = "aura-72h-extended"
        elif total_mem_gb >= 40:
            preset_name = "aura-large"
        else:
            preset_name = "aura-medium"
        print(
            f"Preset automático seleccionado: {preset_name} (GPU {total_mem_gb:.1f} GB)",
            flush=True,
        )
    elif preset_name != "manual" and preset_name not in MODEL_PRESETS:
        raise ValueError(
            f"Preset desconocido '{preset_name}'. Opciones: {sorted(MODEL_PRESETS)}"
        )

    if preset_name != "manual":
        preset = MODEL_PRESETS[preset_name]
        args.embed_size = preset["embed_size"]
        args.num_layers = preset["num_layers"]
        args.num_heads = preset["num_heads"]
        args.seq_length = preset["seq_length"]
    else:
        missing = [
            name
            for name, value in {
                "embed_size": args.embed_size,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "seq_length": args.seq_length,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(
                "Preset 'manual' requiere especificar: " + ", ".join(missing)
            )

    args.dropout = 0.1 if args.dropout is None else float(args.dropout)
    args.embed_size = int(args.embed_size)
    args.num_layers = int(args.num_layers)
    args.num_heads = int(args.num_heads)
    args.seq_length = int(args.seq_length)

    if args.embed_size % args.num_heads != 0:
        raise ValueError(
            "embed_size debe ser divisible entre num_heads (revisa el preset seleccionado)"
        )

    args.selected_preset = preset_name


def resolve_training_hparams(args: argparse.Namespace) -> None:
    preset = getattr(args, "selected_preset", args.model_preset)
    if args.batch_size is None:
        if preset == "aura-72h-max":
            args.batch_size = 2
        elif preset in {"aura-72h-extended", "aura-large"}:
            args.batch_size = 4
        else:
            args.batch_size = 8
    if args.gradient_accumulation_steps is None:
        if preset == "aura-72h-max":
            args.gradient_accumulation_steps = 16
        elif preset in {"aura-72h-extended", "aura-large"}:
            args.gradient_accumulation_steps = 8
        else:
            args.gradient_accumulation_steps = 4
    if args.learning_rate is None:
        if preset == "aura-72h-max":
            args.learning_rate = 2e-4
        elif preset in {"aura-72h-extended", "aura-large"}:
            args.learning_rate = 2.5e-4
        else:
            args.learning_rate = 3e-4
    if args.save_steps is None:
        args.save_steps = 2500 if preset in {"aura-72h-max", "aura-72h-extended"} else 1000

    args.batch_size = max(1, int(args.batch_size))
    args.gradient_accumulation_steps = max(1, int(args.gradient_accumulation_steps))
    args.learning_rate = float(args.learning_rate)
    args.save_steps = max(1, int(args.save_steps))

    if args.grad_clip is not None and args.grad_clip <= 0:
        args.grad_clip = None

    if args.nan_lr_factor <= 0 or args.nan_lr_factor >= 1:
        if args.nan_behavior == "reduce-lr":
            print(
                "[ADVERTENCIA] --nan_lr_factor debe estar entre 0 y 1. "
                "Se usará el valor por defecto 0.5.",
                file=sys.stderr,
            )
        args.nan_lr_factor = 0.5

    if args.nan_min_lr is not None and args.nan_min_lr <= 0:
        if args.nan_behavior == "reduce-lr":
            print(
                "[ADVERTENCIA] --nan_min_lr debe ser mayor que 0. "
                "Se usará el valor por defecto 1e-7.",
                file=sys.stderr,
            )
        args.nan_min_lr = NAN_MIN_LR_DEFAULT

    args.nan_patience = max(1, int(args.nan_patience))

    if args.nan_behavior != "reduce-lr":
        args.nan_min_lr = max(args.nan_min_lr, 0.0)

    if args.gradient_checkpointing is None:
        if preset == "aura-72h-max":
            args.gradient_checkpointing = True
            suggested_segments = max(4, args.num_layers // 2)
            args.checkpoint_segments = suggested_segments
            print(
                "Gradient checkpointing activado automáticamente para el preset "
                "'aura-72h-max' para evitar OOM (usa --no_gradient_checkpointing si deseas "
                "deshabilitarlo)",
                flush=True,
            )
        else:
            args.gradient_checkpointing = False

    if args.gradient_checkpointing:
        args.checkpoint_segments = max(1, int(args.checkpoint_segments))


def resolve_training_precision(args: argparse.Namespace, device: torch.device) -> None:
    requested = (args.precision or "auto").lower()
    resolved = requested

    if requested == "auto":
        if device.type == "cuda":
            is_bf16_supported = bool(
                getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            )
            resolved = "bf16" if is_bf16_supported else "fp16"
        else:
            resolved = "fp32"
    elif requested == "bf16":
        if device.type != "cuda":
            print(
                "[ADVERTENCIA] bf16 no es compatible con el dispositivo actual; "
                "se utilizará fp32.",
                flush=True,
            )
            resolved = "fp32"
        elif not getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            print(
                "[ADVERTENCIA] La GPU no soporta bf16 nativo; se utilizará fp16.",
                flush=True,
            )
            resolved = "fp16"
    elif requested == "fp16" and device.type != "cuda":
        print(
            "[ADVERTENCIA] fp16 requiere GPU; se utilizará fp32.",
            flush=True,
        )
        resolved = "fp32"

    autocast_dtype = None
    if device.type == "cuda" and resolved in {"fp16", "bf16"}:
        autocast_dtype = torch.float16 if resolved == "fp16" else torch.bfloat16

    args.precision = resolved
    args.autocast_dtype = autocast_dtype
    args.use_autocast = autocast_dtype is not None
    args.use_grad_scaler = device.type == "cuda" and resolved == "fp16"


def optimize_for_runtime(
    args: argparse.Namespace,
    device: torch.device,
    train_tokens: int,
    target_hours: float = 36.0,
) -> Dict[str, float]:
    """Ajusta batch, acumulación y checkpointing para cumplir el objetivo temporal."""

    runtime_info: Dict[str, float] = {
        "target_hours": float(max(target_hours, 1e-3)),
        "device_type": device.type,
    }

    if target_hours <= 0:
        target_hours = 36.0

    required_tokens_per_sec = (
        train_tokens / (target_hours * 3600)
        if target_hours > 0 and train_tokens > 0
        else 0.0
    )
    runtime_info["required_tokens_per_sec"] = required_tokens_per_sec

    if device.type != "cuda":
        return runtime_info

    # Medir VRAM libre antes de decidir batch/accum.
    with torch.cuda.device(device):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / GIB
    total_gb = total_bytes / GIB
    runtime_info["gpu_free_gb"] = round(free_gb, 2)
    runtime_info["gpu_total_gb"] = round(total_gb, 2)

    base_batch = max(1, int(args.batch_size))
    base_accum = max(1, int(args.gradient_accumulation_steps))

    print(
        f"Memoria libre detectada: {free_gb:.2f} GiB de {total_gb:.2f} GiB totales",
        flush=True,
    )

    # Ajustar el batch efectivo deseado según la ventana de memoria libre.
    target_effective = 144.0
    if free_gb < 60:
        target_effective = 112.0
    elif free_gb > 70:
        target_effective = 160.0

    max_micro_batch = base_batch
    if free_gb > 0:
        max_micro_batch = max(
            base_batch,
            min(16, max(1, int(free_gb // 8))),
        )
    if free_gb > 70:
        max_micro_batch = max(
            max_micro_batch,
            base_batch + int(max(1.0, (free_gb - 70) / 2.0)),
        )

    desired_batch = min(
        max_micro_batch,
        max(base_batch, int(math.ceil(target_effective / base_accum))),
    )
    proposed_batch = max(1, desired_batch)

    proposed_grad_accum = max(1, int(round(target_effective / proposed_batch)))

    if free_gb < 60:
        proposed_grad_accum = max(1, min(proposed_grad_accum, max(4, base_accum // 2)))

    effective = proposed_batch * proposed_grad_accum
    if free_gb >= 60:
        if effective < 128:
            proposed_grad_accum = math.ceil(128 / proposed_batch)
        elif effective > 160:
            proposed_grad_accum = max(1, math.floor(160 / proposed_batch))
    else:
        if effective > 112:
            proposed_grad_accum = max(1, math.floor(112 / proposed_batch))

    proposed_grad_accum = max(1, proposed_grad_accum)
    effective = proposed_batch * proposed_grad_accum

    if free_gb > 70:
        proposed_grad_accum = max(proposed_grad_accum, base_accum)
        effective = proposed_batch * proposed_grad_accum
        if effective > 160:
            proposed_grad_accum = max(1, math.floor(160 / proposed_batch))
            effective = proposed_batch * proposed_grad_accum

    # Reducir segmentos de checkpointing si la VRAM extra lo permite.
    if args.gradient_checkpointing:
        if free_gb > 70:
            target_segments = max(2, args.num_layers // 4)
        elif free_gb > 60:
            target_segments = max(4, args.num_layers // 3)
        else:
            target_segments = args.checkpoint_segments
        if target_segments < args.checkpoint_segments:
            print(
                "Reduciendo checkpoint_segments para aprovechar más VRAM: "
                f"{args.checkpoint_segments} -> {target_segments}",
                flush=True,
            )
            args.checkpoint_segments = target_segments

    adjustments = []
    if proposed_batch != base_batch:
        adjustments.append(f"batch_size {base_batch} -> {proposed_batch}")
    if proposed_grad_accum != base_accum:
        adjustments.append(
            f"gradient_accumulation_steps {base_accum} -> {proposed_grad_accum}"
        )

    args.batch_size = proposed_batch
    args.gradient_accumulation_steps = proposed_grad_accum
    args.save_steps = max(2500, int(args.save_steps))

    effective = args.batch_size * args.gradient_accumulation_steps
    tokens_per_step = effective * args.seq_length
    runtime_info["effective_batch"] = effective
    runtime_info["tokens_per_step"] = tokens_per_step

    estimated_steps = math.ceil(train_tokens / max(tokens_per_step, 1))
    runtime_info["estimated_optimizer_steps"] = estimated_steps

    if required_tokens_per_sec > 0:
        runtime_info["required_step_time"] = tokens_per_step / required_tokens_per_sec

    if adjustments:
        print(
            "Optimizaciones de runtime aplicadas: " + ", ".join(adjustments),
            flush=True,
        )
    else:
        print("Optimizaciones de runtime: configuración base conservada", flush=True)

    print(
        "Objetivo de throughput: "
        f"{required_tokens_per_sec:,.0f} tok/s para cumplir {target_hours:.1f} h",
        flush=True,
    )

    return runtime_info


def finalize_warmup(args: argparse.Namespace, total_optimizer_steps: int) -> None:
    if args.warmup_steps is None:
        computed = max(10, int(0.03 * total_optimizer_steps))
        args.warmup_steps = min(computed, max(total_optimizer_steps // 2, 10))


def load_tokenizer(path_or_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(path_or_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token or "<|pad|>"})
    tokenizer.model_max_length = int(1e12)  # sin truncamiento implícito
    return tokenizer


def resolve_dataset_file(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted([*path.glob("*.txt"), *path.glob("*.jsonl")])
        if not candidates:
            raise FileNotFoundError(
                f"No se encontraron archivos .txt o .jsonl en {path}."
            )
        selected = candidates[0]
        print(
            f"Se utilizará el corpus {selected.name} ubicado en {selected.parent}",
            flush=True,
        )
        return selected
    raise FileNotFoundError(f"La ruta {path} no existe")


def extract_text_field(example: Dict[str, object], field: Optional[str]) -> Optional[str]:
    if field and field in example:
        value = example[field]
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            value = " ".join(str(v) for v in value if v)
        return str(value).strip()

    for candidate in ("text", "content", "body", "sentence", "tweet", "tweet_text"):
        if candidate in example and example[candidate]:
            value = example[candidate]
            if isinstance(value, (list, tuple)):
                value = " ".join(str(v) for v in value if v)
            return str(value).strip()
    return None


def download_hf_corpus(
    args: argparse.Namespace,
) -> Path:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependencia externa
        raise ImportError(
            "Para descargar datasets automáticamente instala `datasets` (pip install datasets)."
        ) from exc

    dataset_name = args.hf_dataset_name
    if not dataset_name or dataset_name.lower() in {"none", "null"}:
        raise FileNotFoundError(
            "dataset_path no existe y no se proporcionó hf_dataset_name para descargar automáticamente."
        )

    target = args.dataset_path
    if target.suffix:
        output_dir = target.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = target
    else:
        output_dir = target
        output_dir.mkdir(parents=True, exist_ok=True)
        sanitized = dataset_name.replace("/", "_")
        config_suffix = args.hf_dataset_config or "default"
        output_file = output_dir / f"{sanitized}-{config_suffix}-{args.hf_dataset_split}.jsonl"

    config_display = args.hf_dataset_config or "-"
    print(
        "Descargando dataset desde Hugging Face: "
        f"{dataset_name} ({config_display}/{args.hf_dataset_split})",
        flush=True,
    )

    try:
        dataset = load_dataset(
            dataset_name,
            args.hf_dataset_config or None,
            split=args.hf_dataset_split,
            streaming=args.hf_streaming,
            use_auth_token=args.hf_auth_token,
            trust_remote_code=args.hf_trust_remote_code,
        )
    except Exception as exc:
        hint = [
            f"No se pudo descargar el dataset '{dataset_name}'.",
            "Verifica que el nombre/configuración/split existan y, si es un dataset con acceso restringido,",
            "proporciona un token con --hf_auth_token o inicia sesión con huggingface-cli login.",
        ]
        message = str(exc)
        if "Dataset scripts are no longer supported" in message or "trust_remote_code" in message:
            hint.append(
                "Si el dataset requiere código remoto, añade --hf_trust_remote_code o actívalo en el preset."
            )
        raise RuntimeError(" ".join(hint)) from exc

    import json as _json

    written_examples = 0
    with output_file.open("w", encoding="utf-8") as f:
        iterator: Iterable[Dict[str, object]]
        iterator = dataset if args.hf_streaming else dataset  # type: ignore[assignment]
        for example in iterator:
            text = extract_text_field(example, args.hf_text_field)
            if not text:
                continue
            record = {"text": text}
            f.write(_json.dumps(record, ensure_ascii=False) + "\n")
            written_examples += 1
            if args.hf_download_limit and written_examples >= args.hf_download_limit:
                break

    if written_examples == 0:
        raise ValueError(
            "No se pudieron extraer ejemplos del dataset descargado. Revisa hf_text_field o dataset configurado."
        )

    print(
        f"Dataset guardado en {output_file} con {written_examples:,} ejemplos",
        flush=True,
    )
    return output_file


def ensure_dataset_file(args: argparse.Namespace) -> Path:
    path = args.dataset_path
    if path.exists():
        try:
            return resolve_dataset_file(path)
        except FileNotFoundError as exc:
            if args.skip_auto_download:
                raise
            print(
                "No se encontró un corpus utilizable en la ruta especificada; "
                "se procederá a descargarlo automáticamente.",
                flush=True,
            )
            return download_hf_corpus(args)
    if args.skip_auto_download:
        raise FileNotFoundError(
            f"La ruta {path} no existe y se solicitó omitir la descarga automática"
        )
    return download_hf_corpus(args)


def train_tokenizer_from_corpus(
    dataset_path: Path,
    vocab_size: int,
    min_frequency: int,
    output_dir: Path,
) -> PreTrainedTokenizerBase:
    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:  # pragma: no cover - dependencia opcional
        raise ImportError(
            "Para entrenar el tokenizer automáticamente instala `tokenizers` (pip install tokenizers)."
        ) from exc

    print(
        "Entrenando tokenizer ByteLevel BPE directamente desde el corpus…",
        flush=True,
    )

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
    )
    tokenizer.train_from_iterator(read_corpus(dataset_path), trainer=trainer)

    from transformers import PreTrainedTokenizerFast

    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    hf_tokenizer.pad_token = "<|pad|>"
    hf_tokenizer.bos_token = "<|bos|>"
    hf_tokenizer.eos_token = "<|eos|>"
    hf_tokenizer.unk_token = "<|unk|>"
    hf_tokenizer.model_max_length = int(1e12)
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_tokenizer.save_pretrained(output_dir)
    print(
        f"Tokenizer entrenado y guardado en {output_dir} (vocab_size={len(hf_tokenizer)})",
        flush=True,
    )
    return hf_tokenizer


def prepare_tokenizer(
    args: argparse.Namespace, dataset_path: Path
) -> Tuple[PreTrainedTokenizerBase, Dict[str, object]]:
    if args.tokenizer_name_or_path.lower() == "auto":
        tokenizer_dir = args.output_dir / "tokenizer"
        reused = False
        if tokenizer_dir.exists():
            try:
                tokenizer = load_tokenizer(str(tokenizer_dir))
                reused = True
                print(
                    f"Reutilizando tokenizer existente en {tokenizer_dir}",
                    flush=True,
                )
                return tokenizer, {
                    "tokenizer_source": "auto",
                    "tokenizer_dir": str(tokenizer_dir),
                    "tokenizer_reused": True,
                }
            except Exception as exc:
                print(
                    "No se pudo reutilizar el tokenizer existente. "
                    "Se volverá a entrenar desde el corpus.",
                    flush=True,
                )
                if os.environ.get("AURA_DEBUG"):
                    print(f"Motivo del fallo al cargar tokenizer: {exc}")
        tokenizer = train_tokenizer_from_corpus(
            dataset_path,
            args.tokenizer_vocab_size,
            args.tokenizer_min_frequency,
            tokenizer_dir,
        )
        return tokenizer, {
            "tokenizer_source": "auto",
            "tokenizer_dir": str(tokenizer_dir),
            "tokenizer_reused": reused,
        }

    tokenizer = load_tokenizer(args.tokenizer_name_or_path)
    saved_dir = args.output_dir / "tokenizer"
    tokenizer.save_pretrained(saved_dir)
    return tokenizer, {
        "tokenizer_source": args.tokenizer_name_or_path,
        "tokenizer_dir": str(saved_dir),
        "tokenizer_reused": False,
    }


def read_corpus(path: Path) -> Iterable[str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as f:
            text = f.read()
        yield text
    elif suffix == ".jsonl":
        import json as _json

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = _json.loads(line)
                if "text" not in sample:
                    raise ValueError("Las entradas .jsonl deben contener una clave 'text'")
                yield sample["text"]
    else:
        raise ValueError(f"Formato no soportado: {suffix}")


def tokenize_corpus(
    tokenizer: PreTrainedTokenizerBase,
    corpus: Iterable[str],
) -> torch.Tensor:
    env_chunk_size = os.environ.get("AURA_TOKENIZE_CHUNK_SIZE")
    if env_chunk_size is not None:
        try:
            chunk_size = int(env_chunk_size)
        except ValueError:
            print(
                "Valor inválido para AURA_TOKENIZE_CHUNK_SIZE; usando el tamaño por defecto",
                flush=True,
            )
            chunk_size = TOKENIZE_CHUNK_SIZE
    else:
        chunk_size = TOKENIZE_CHUNK_SIZE
    chunk_size = max(chunk_size, 1)

    tmp_path: Optional[Path] = None
    binary_file: Optional[BinaryIO] = None
    in_memory_tokens: Optional[array] = None
    total_tokens = 0
    total_lines = 0
    chunk: list[str] = []

    if _TOKEN_CACHE_PATH is not None:
        tmp_path = _TOKEN_CACHE_PATH.with_suffix(".tmp_tokens.bin")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        binary_file = tmp_path.open("wb")
    else:
        in_memory_tokens = array("q")

    def process_chunk(lines: list[str]) -> None:
        nonlocal total_lines, total_tokens
        if not lines:
            return

        batch_encoding = tokenizer(
            lines,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            padding=False,
            truncation=False,
        )

        input_ids = batch_encoding.get("input_ids", [])
        if isinstance(input_ids, torch.Tensor):
            sequences = input_ids.tolist()
        else:
            sequences = input_ids

        if isinstance(sequences, list) and sequences and isinstance(sequences[0], int):
            sequences = [sequences]  # type: ignore[assignment]

        chunk_tokens = array("q")
        for ids in sequences or []:
            if not ids:
                continue
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            elif isinstance(ids, int):
                ids = [ids]
            chunk_tokens.extend(int(token_id) for token_id in ids)

        produced_tokens = len(chunk_tokens)
        total_lines += len(lines)
        total_tokens += produced_tokens

        if produced_tokens == 0:
            return

        if binary_file is not None:
            chunk_tokens.tofile(binary_file)
        else:
            assert in_memory_tokens is not None
            in_memory_tokens.extend(chunk_tokens)

        print(
            f"Procesadas {total_lines:,} líneas... Tokens acumulados: {total_tokens:,}",
            flush=True,
        )

    try:
        for text in corpus:
            if text is None:
                continue
            if not isinstance(text, str):
                text = str(text)
            if not text:
                continue
            chunk.append(text)
            if len(chunk) >= chunk_size:
                process_chunk(chunk)
                chunk.clear()

        process_chunk(chunk)
    finally:
        if binary_file is not None:
            binary_file.close()

    if total_tokens == 0:
        raise ValueError("El corpus tokenizado quedó vacío")

    if tmp_path is not None and tmp_path.exists():
        np_tokens = np.fromfile(tmp_path, dtype=np.int64)
        token_ids = torch.from_numpy(np_tokens)
        tmp_path.unlink(missing_ok=True)
    else:
        assert in_memory_tokens is not None
        token_ids = torch.tensor(in_memory_tokens, dtype=torch.long)

    if _TOKEN_CACHE_PATH is not None:
        _TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(token_ids, _TOKEN_CACHE_PATH)
        print(
            f"Tokens tokenizados guardados en {_TOKEN_CACHE_PATH}",
            flush=True,
        )

    print(
        f"Tokenización completada. Total líneas: {total_lines:,}. Tokens: {token_ids.numel():,}",
        flush=True,
    )
    return token_ids


def _limit_tokens(
    token_slice: torch.Tensor, seq_length: int, limit: Optional[int]
) -> torch.Tensor:
    if limit is None or limit <= 0 or limit >= token_slice.numel():
        return token_slice
    if limit < seq_length + 1:
        return token_slice
    trimmed = (limit // seq_length) * seq_length
    if trimmed <= seq_length:
        return token_slice
    max_available = max(0, token_slice.numel() - 1)
    trimmed = min(trimmed, max_available)
    keep = max(seq_length + 1, trimmed + 1)
    return token_slice[:keep]


def prepare_datasets(
    token_ids: torch.Tensor,
    seq_length: int,
    validation_split: float,
    train_token_limit: Optional[int] = None,
    val_token_limit: Optional[int] = None,
) -> Tuple[SpanishCorpus, Optional[SpanishCorpus]]:
    if len(token_ids) <= seq_length:
        raise ValueError("El corpus es demasiado pequeño para el seq_length especificado")

    if validation_split <= 0 or validation_split >= 1:
        train_slice = _limit_tokens(token_ids, seq_length, train_token_limit)
        train_dataset = SpanishCorpus(train_slice, seq_length)
        return train_dataset, None

    val_tokens = int(len(token_ids) * validation_split)
    val_tokens = max(seq_length + 1, val_tokens)
    train_tokens = len(token_ids) - val_tokens
    if train_tokens <= seq_length:
        raise ValueError("validation_split demasiado grande para el tamaño del corpus")

    train_ids = _limit_tokens(token_ids[:train_tokens], seq_length, train_token_limit)
    val_ids = _limit_tokens(token_ids[train_tokens:], seq_length, val_token_limit)
    return SpanishCorpus(train_ids, seq_length), SpanishCorpus(val_ids, seq_length)


def create_model(
    vocab_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[GPT2, ModelConfig]:
    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_length=args.seq_length,
        embed_size=args.embed_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = GPT2(config)
    model.to(device)

    if args.torch_compile:
        if not hasattr(torch, "compile"):
            print(
                "torch.compile no está disponible en esta versión de PyTorch; "
                "continuando sin compilar.",
                flush=True,
            )
        else:
            compile_kwargs = {"mode": args.compile_mode}
            if args.compile_fullgraph:
                compile_kwargs["fullgraph"] = True
            try:
                model = torch.compile(model, **compile_kwargs)
                print(
                    "Modelo compilado con torch.compile "
                    f"(modo={args.compile_mode}, fullgraph={args.compile_fullgraph}).",
                    flush=True,
                )
            except Exception as compile_error:  # pragma: no cover - dependiente del backend
                print(
                    "No se pudo compilar el modelo con torch.compile: "
                    f"{compile_error}. Se continúa sin compilación.",
                    flush=True,
                )

    return model, config


def create_optimizer(model: GPT2, lr: float, weight_decay: float):
    try:
        import bitsandbytes as bnb  # type: ignore

        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
        print("Usando AdamW 8-bit de bitsandbytes", flush=True)
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
        print("bitsandbytes no disponible, usando torch.optim.AdamW", flush=True)
    return optimizer


def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def save_checkpoint(
    output_dir: Path,
    model: GPT2,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    tokenizer: PreTrainedTokenizerBase,
    step: int,
    epoch: int,
    config: ModelConfig,
    tag: str,
    scheduler_state: Optional[dict] = None,
    extra_metadata: Optional[Dict[str, object]] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint-{tag}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "global_step": step,
            "epoch": epoch,
            "config": config.__dict__,
            "scheduler_state_dict": scheduler_state,
            "extra_metadata": extra_metadata,
        },
        checkpoint_path,
    )
    tokenizer.save_pretrained(output_dir)
    print(f"Checkpoint guardado en {checkpoint_path}", flush=True)
    return checkpoint_path


def save_training_log(
    log_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not log_path.exists()
    with log_path.open("a", encoding="utf-8") as f:
        if header_needed:
            f.write("epoch,train_loss,val_loss\n")
        val_text = "" if val_loss is None else f"{val_loss:.6f}"
        f.write(f"{epoch},{train_loss:.6f},{val_text}\n")


def enable_gradient_checkpointing(model: GPT2, segments: int) -> None:
    segments = max(1, segments)

    def forward_with_checkpoint(input_token: torch.Tensor) -> torch.Tensor:
        batch, seq_length = input_token.size()
        device = input_token.device
        pos = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)
        x = model.token_embed(input_token) + model.pos_embed(pos)
        x = model.dropout(x)
        x = checkpoint_sequential(model.transformers, segments, x)
        x = model.norml(x)
        return x @ model.token_embed.weight.t()

    model.forward = forward_with_checkpoint  # type: ignore[assignment]


def evaluate(
    model: GPT2,
    dataloader: DataLoader,
    device: torch.device,
    use_autocast: bool = False,
    autocast_dtype: Optional[torch.dtype] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            amp_context = (
                autocast(dtype=autocast_dtype) if use_autocast else nullcontext()
            )
            with amp_context:
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
            total_loss += loss.item()
            total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_mem_gb = 0.0
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        total_mem_gb = properties.total_memory / (1024**3)
        print(
            f"Usando GPU {properties.name} con {total_mem_gb:.1f} GB", flush=True
        )
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        print(f"Memoria libre tras limpieza: {torch.cuda.mem_get_info()[0] / (1024**3):.2f} GB")

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("medium")  # type: ignore[attr-defined]
    else:
        print("Advertencia: no se detectó GPU, el entrenamiento será muy lento", flush=True)

    dataset_file = ensure_dataset_file(args)
    tokenizer, tokenizer_info = prepare_tokenizer(args, dataset_file)
    tokenizer_info["tokenizer_vocab_size"] = len(tokenizer)
    tokenizer_info["tokenizer_pad_token"] = tokenizer.pad_token
    apply_model_preset(args, total_mem_gb)
    resolve_training_hparams(args)
    resolve_training_precision(args, device)

    global _TOKEN_CACHE_PATH
    _TOKEN_CACHE_PATH = args.output_dir / TOKEN_CACHE_FILENAME

    if _TOKEN_CACHE_PATH.exists():
        print(f"Cargando tokens preprocesados desde {_TOKEN_CACHE_PATH}", flush=True)
        token_ids = torch.load(_TOKEN_CACHE_PATH, map_location="cpu")
        if not isinstance(token_ids, torch.Tensor):
            raise ValueError(
                "El archivo de tokens guardado no contiene un tensor válido. Elimínalo y vuelve a tokenizar."
            )
        token_ids = token_ids.long().contiguous()
        print(
            f"Tokens cargados: {token_ids.numel():,}. Se omite la tokenización.",
            flush=True,
        )
    else:
        token_ids = tokenize_corpus(tokenizer, read_corpus(dataset_file))
    train_dataset, val_dataset = prepare_datasets(
        token_ids,
        args.seq_length,
        args.validation_split,
        train_token_limit=args.limit_train_tokens,
        val_token_limit=args.limit_val_tokens,
    )

    train_tokens = getattr(train_dataset, "effective_token_count", len(train_dataset.text))
    val_tokens = (
        getattr(val_dataset, "effective_token_count", len(val_dataset.text))
        if val_dataset is not None
        else 0
    )
    total_tokens = train_tokens + val_tokens
    print(
        f"Tokens disponibles -> entrenamiento: {train_tokens:,} | validación: {val_tokens:,} | total: {total_tokens:,}",
        flush=True,
    )
    if args.limit_train_tokens is not None:
        print(
            "    Límite aplicado en entrenamiento: "
            f"{args.limit_train_tokens:,} tokens solicitados",
            flush=True,
        )
    if args.limit_val_tokens is not None and val_dataset is not None:
        print(
            "    Límite aplicado en validación: "
            f"{args.limit_val_tokens:,} tokens solicitados",
            flush=True,
        )

    # Ajuste automático de parámetros para cumplir el presupuesto temporal.
    runtime_info = optimize_for_runtime(
        args,
        device,
        train_tokens=train_tokens,
        target_hours=args.target_hours,
    )

    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0 and args.prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = max(1, args.prefetch_factor)

    train_loader = DataLoader(train_dataset, **dataloader_kwargs)

    val_loader = None
    if val_dataset is not None:
        val_kwargs = dataloader_kwargs.copy()
        val_kwargs["shuffle"] = False
        val_loader = DataLoader(val_dataset, **val_kwargs)

    micro_batches_per_epoch = len(train_loader)
    if micro_batches_per_epoch == 0:
        raise ValueError("No hay batches disponibles. Reduce batch_size o revisa el corpus.")

    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / args.gradient_accumulation_steps
    )
    total_optimizer_steps = optimizer_steps_per_epoch * args.epochs
    runtime_info["optimizer_steps_per_epoch"] = optimizer_steps_per_epoch
    runtime_info["micro_batches_per_epoch"] = micro_batches_per_epoch
    finalize_warmup(args, total_optimizer_steps)

    effective_batch = args.batch_size * args.gradient_accumulation_steps
    tokens_per_step = args.seq_length * effective_batch
    print(
        "Configuración de modelo "
        f"[{args.selected_preset}]: d_model={args.embed_size}, capas={args.num_layers}, "
        f"heads={args.num_heads}, contexto={args.seq_length}",
        flush=True,
    )
    print(
        f"Batch real={args.batch_size} | acumulación={args.gradient_accumulation_steps} | "
        f"batch efectivo={effective_batch} | tokens/step={tokens_per_step:,}",
        flush=True,
    )
    print(
        f"Pasos de optimización/época={optimizer_steps_per_epoch} (microbatches={micro_batches_per_epoch}), "
        f"total pasos={total_optimizer_steps}, warmup={args.warmup_steps}",
        flush=True,
    )
    required_step_time = runtime_info.get("required_step_time")
    if required_step_time:
        print(
            f"Tiempo objetivo por paso: {required_step_time:.2f} s para completar en {args.target_hours:.1f} h",
            flush=True,
        )

    model, config = create_model(len(tokenizer), args, device)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model, args.checkpoint_segments)
        print(
            f"Gradient checkpointing activado con {args.checkpoint_segments} segmentos",
            flush=True,
        )

    if args.use_autocast:
        precision_label = f"autocast {str(args.autocast_dtype).split('.')[-1]}"
    else:
        precision_label = args.precision
    print(f"Modo de precisión: {precision_label}", flush=True)

    model.train()
    if device.type == "cuda":
        # Reiniciar contadores para reportar el pico de VRAM del entrenamiento.
        torch.cuda.reset_peak_memory_stats(device)
    max_vram_gb = 0.0
    num_params = sum(p.numel() for p in model.parameters())
    print(
        f"Parámetros del modelo: {num_params/1e9:.2f}B ({num_params:,} parámetros)",
        flush=True,
    )

    optimizer = create_optimizer(model, args.learning_rate, args.weight_decay)
    scaler = GradScaler(enabled=args.use_grad_scaler)

    scheduler = None
    if args.warmup_steps and args.warmup_steps > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_optimizer_steps,
        )
        print(
            f"Scheduler lineal con warmup de {args.warmup_steps} pasos (total {total_optimizer_steps})",
            flush=True,
        )
    else:
        print("Warmup desactivado", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training_log.csv"

    start_epoch = 0
    global_step = 0

    restored_metadata: Dict[str, object] = {}

    if args.resume_from is not None:
        print(f"Reanudando desde {args.resume_from}", flush=True)
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler.is_enabled():
            if scaler_state:
                scaler.load_state_dict(scaler_state)
            else:
                print(
                    "[AVISO] El checkpoint no contenía estado del GradScaler; se reiniciará.",
                    flush=True,
                )
        elif scaler_state:
            print(
                "[AVISO] Se ignoró el estado del GradScaler del checkpoint porque "
                "la precisión actual no usa escalado.",
                flush=True,
            )
        global_step = checkpoint.get("global_step", 0)
        start_epoch = checkpoint.get("epoch", 0)
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        restored_metadata = checkpoint.get("extra_metadata") or {}
        if restored_metadata:
            print(f"Metadatos restaurados: {restored_metadata}", flush=True)
            if args.nan_behavior == "reduce-lr":
                restored_min_lr = restored_metadata.get("nan_min_lr")
                try:
                    restored_min_lr_value = float(restored_min_lr)
                except (TypeError, ValueError):
                    restored_min_lr_value = None
                if (
                    restored_min_lr_value is not None
                    and math.isfinite(restored_min_lr_value)
                    and math.isclose(
                        args.nan_min_lr,
                        NAN_MIN_LR_DEFAULT,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    args.nan_min_lr = max(
                        restored_min_lr_value,
                        NAN_MIN_LR_DEFAULT,
                    )
            restored_patience = restored_metadata.get("nan_patience")
            if restored_patience is not None:
                try:
                    restored_patience_int = max(1, int(restored_patience))
                except (TypeError, ValueError):
                    restored_patience_int = None
                if (
                    restored_patience_int is not None
                    and args.nan_patience == NAN_PATIENCE_DEFAULT
                ):
                    args.nan_patience = restored_patience_int
            restored_precision = restored_metadata.get("precision")
            if (
                isinstance(restored_precision, str)
                and restored_precision
                and restored_precision != args.precision
            ):
                print(
                    "[AVISO] El checkpoint se entrenó con precisión "
                    f"{restored_precision}; precisión actual: {args.precision}.",
                    flush=True,
                )
        model.train()

    initial_global_step = global_step

    stop_training = False
    checkpoint_metadata = {
        "model_preset": args.selected_preset,
        "gradient_checkpointing": args.gradient_checkpointing,
        "effective_batch_size": effective_batch,
        "tokens_per_step": tokens_per_step,
        "dataset_file": str(dataset_file),
        "hf_dataset_name": args.hf_dataset_name,
        "hf_dataset_config": args.hf_dataset_config,
        "hf_dataset_split": args.hf_dataset_split,
        "auto_download": not args.skip_auto_download,
        "nan_behavior": args.nan_behavior,
        "nan_lr_factor": args.nan_lr_factor,
        "nan_min_lr": args.nan_min_lr,
        "nan_patience": args.nan_patience,
        "precision": args.precision,
        "autocast_dtype": str(args.autocast_dtype) if args.autocast_dtype else None,
        "limit_train_tokens": args.limit_train_tokens,
        "limit_val_tokens": args.limit_val_tokens,
        "torch_compile": args.torch_compile,
        "compile_mode": args.compile_mode,
        "compile_fullgraph": args.compile_fullgraph,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        **tokenizer_info,
    }
    checkpoint_metadata["optimizer_steps_per_epoch"] = optimizer_steps_per_epoch
    checkpoint_metadata["micro_batches_per_epoch"] = micro_batches_per_epoch
    for key, value in runtime_info.items():
        checkpoint_metadata[f"runtime_{key}"] = value
    nan_events = int(restored_metadata.get("nan_events", 0))
    nan_triggered_stop = bool(restored_metadata.get("nan_triggered_stop", False))
    last_nan_adjust_step = int(restored_metadata.get("last_nan_adjust_step", -1))
    checkpoint_metadata["nan_events"] = nan_events
    checkpoint_metadata["nan_triggered_stop"] = nan_triggered_stop
    checkpoint_metadata["last_nan_adjust_step"] = last_nan_adjust_step

    def handle_interrupt(signum, frame):  # pragma: no cover - señal externa
        nonlocal stop_training
        stop_training = True
        print("Interrupción recibida, se guardará checkpoint al finalizar el paso en curso", flush=True)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    step_start_time = time.time()
    consecutive_nan_batches = 0

    try:
        for epoch in range(start_epoch, args.epochs):
            if stop_training:
                break
            epoch_losses = []
            epoch_start_time = time.time()
            progress_bar = tqdm(train_loader, desc=f"Época {epoch + 1}/{args.epochs}", leave=False)
            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                if stop_training:
                    break
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                amp_context = (
                    autocast(dtype=args.autocast_dtype)
                    if args.use_autocast
                    else nullcontext()
                )
                logits = None
                loss = None
                try:
                    with amp_context:
                        logits = model(inputs)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1),
                        )
                    loss_value = loss.item()
                except RuntimeError as runtime_err:
                    if "out of memory" in str(runtime_err).lower():
                        progress_bar.write(
                            f"[OOM detectado] paso global {global_step}, "
                            f"época {epoch + 1}, batch {batch_idx}."
                        )
                        optimizer.zero_grad(set_to_none=True)
                        micro_step = 0
                        step_start_time = time.time()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        del inputs, labels
                        if logits is not None:
                            del logits
                        if loss is not None:
                            del loss
                        continue
                    raise

                if not math.isfinite(loss_value):
                    nan_events += 1
                    consecutive_nan_batches += 1
                    checkpoint_metadata["nan_events"] = nan_events
                    message = (
                        f"[NaN detectado] paso global {global_step}, "
                        f"época {epoch + 1}, batch {batch_idx}."
                    )
                    progress_bar.write(message)
                    optimizer.zero_grad(set_to_none=True)
                    micro_step = 0
                    step_start_time = time.time()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    if args.nan_behavior == "reduce-lr":
                        if last_nan_adjust_step == global_step:
                            current_lr = optimizer.param_groups[0]["lr"]
                            progress_bar.write(
                                f"    LR ya ajustada en este paso (actual {current_lr:.2e})."
                            )
                        else:
                            lr_changes = []
                            adjustment_applied = False
                            for group in optimizer.param_groups:
                                old_lr = float(group.get("lr", 0.0))
                                target_lr = old_lr * args.nan_lr_factor
                                new_lr = max(target_lr, args.nan_min_lr)
                                if not math.isfinite(new_lr):
                                    new_lr = args.nan_min_lr
                                if "initial_lr" in group:
                                    group["initial_lr"] = new_lr
                                group["lr"] = new_lr
                                lr_changes.append((old_lr, new_lr))
                                if not math.isclose(
                                    old_lr,
                                    new_lr,
                                    rel_tol=1e-6,
                                    abs_tol=max(args.nan_min_lr, 1e-12),
                                ):
                                    adjustment_applied = True
                            if scheduler is not None and hasattr(scheduler, "base_lrs"):
                                scheduler.base_lrs = [
                                    group["lr"] for group in optimizer.param_groups
                                ]
                            if adjustment_applied:
                                progress_bar.write(
                                    "    LR reducida tras NaN: "
                                    + ", ".join(
                                        f"{old:.2e}->{new:.2e}" for old, new in lr_changes
                                    )
                                )
                            else:
                                progress_bar.write(
                                    f"    La LR ya está en el mínimo permitido ({args.nan_min_lr:.2e})."
                                )
                            last_nan_adjust_step = global_step
                            checkpoint_metadata["last_nan_adjust_step"] = (
                                last_nan_adjust_step
                            )
                    elif args.nan_behavior == "stop":
                        stop_training = True
                    else:
                        progress_bar.write(
                            "    Batch con NaN descartado; se continuará."
                        )

                    terminate_training = stop_training
                    if (
                        not terminate_training
                        and args.nan_behavior != "stop"
                        and consecutive_nan_batches >= args.nan_patience
                    ):
                        progress_bar.write(
                            "    Se alcanzó el límite de reintentos tras NaN; deteniendo entrenamiento."
                        )
                        terminate_training = True

                    if terminate_training:
                        stop_training = True
                        nan_triggered_stop = True
                        checkpoint_metadata["nan_triggered_stop"] = True
                        if logits is not None:
                            del logits
                        if loss is not None:
                            del loss
                        del inputs, labels
                        break

                    if logits is not None:
                        del logits
                    if loss is not None:
                        del loss
                    del inputs, labels
                    continue

                consecutive_nan_batches = 0

                loss = loss / args.gradient_accumulation_steps

                scaler.scale(loss).backward()
                micro_step += 1
                epoch_losses.append(loss_value)

                if micro_step == args.gradient_accumulation_steps:
                    if args.grad_clip is not None:
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    micro_step = 0

                    if scheduler is not None:
                        scheduler.step()

                    elapsed = time.time() - step_start_time
                    steps_done = max(global_step - initial_global_step, 1)
                    avg_step_time = elapsed / steps_done
                    remaining = max(total_optimizer_steps - global_step, 0)
                    eta = format_time(avg_step_time * remaining)
                    avg_tokens_per_sec = (
                        tokens_per_step / avg_step_time if avg_step_time > 0 else 0.0
                    )
                    # Calcular métricas online para estimar ETA total/por época.
                    epoch_elapsed = time.time() - epoch_start_time
                    micro_done = batch_idx + 1
                    micro_remaining = max(micro_batches_per_epoch - micro_done, 0)
                    avg_micro_time = epoch_elapsed / max(micro_done, 1)
                    epoch_eta_seconds = avg_micro_time * micro_remaining
                    epoch_eta = format_time(epoch_eta_seconds)
                    current_alloc_gb = 0.0
                    if device.type == "cuda":
                        # Registrar VRAM actual y pico para depuración posterior.
                        with torch.cuda.device(device):
                            current_alloc = torch.cuda.memory_allocated()
                            peak_alloc = torch.cuda.max_memory_allocated()
                        current_alloc_gb = current_alloc / GIB
                        max_vram_gb = max(max_vram_gb, peak_alloc / GIB)
                        checkpoint_metadata["max_vram_gb"] = max_vram_gb
                    checkpoint_metadata["last_tokens_per_sec"] = avg_tokens_per_sec
                    checkpoint_metadata["last_eta_total"] = eta

                    if args.log_interval > 0 and global_step % args.log_interval == 0:
                        current_lr = optimizer.param_groups[0]["lr"]
                        progress_bar.set_postfix(
                            loss=f"{loss_value:.4f}",
                            lr=f"{current_lr:.2e}",
                            eta_total=eta,
                            eta_epoch=epoch_eta,
                            toks=f"{avg_tokens_per_sec:,.0f}",
                            vram=f"{current_alloc_gb:.1f}G",
                        )
                        # Emitir log textual para trazabilidad fuera de la barra de progreso.
                        progress_bar.write(
                            f"[Paso {global_step}] loss={loss_value:.4f} | lr={current_lr:.2e} "
                            f"| tok/s={avg_tokens_per_sec:,.0f} | VRAM={current_alloc_gb:.1f} GiB "
                            f"| ETA total={eta} | ETA época={epoch_eta}"
                        )

                    if args.save_steps and global_step % args.save_steps == 0:
                        save_checkpoint(
                            args.output_dir,
                            model,
                            optimizer,
                            scaler,
                            tokenizer,
                            global_step,
                            epoch,
                            config,
                            f"step-{global_step}",
                            scheduler.state_dict() if scheduler is not None else None,
                            extra_metadata=checkpoint_metadata,
                        )

                    if stop_training:
                        break

            if stop_training and nan_triggered_stop:
                break

            train_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
            if not stop_training and micro_step > 0:
                if args.grad_clip is not None:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if scheduler is not None:
                    scheduler.step()
                if args.save_steps and global_step % args.save_steps == 0:
                    save_checkpoint(
                        args.output_dir,
                        model,
                        optimizer,
                        scaler,
                        tokenizer,
                        global_step,
                        epoch,
                        config,
                        f"step-{global_step}",
                        scheduler.state_dict() if scheduler is not None else None,
                        extra_metadata=checkpoint_metadata,
                    )
                micro_step = 0
            val_loss = None
            if val_loader is not None:
                val_loss = evaluate(
                    model,
                    val_loader,
                    device,
                    use_autocast=args.use_autocast,
                    autocast_dtype=args.autocast_dtype,
                )

            save_training_log(log_path, epoch + 1, train_loss, val_loss)
            print(
                f"Época {epoch + 1} completada. Pérdida entrenamiento: {train_loss:.4f}" +
                (f", pérdida validación: {val_loss:.4f}" if val_loss is not None else ""),
                flush=True,
            )

            if stop_training:
                break

    except KeyboardInterrupt:
        stop_training = True
        print("Entrenamiento interrumpido por el usuario.", flush=True)
    except Exception:
        current_epoch = locals().get("epoch", start_epoch)
        save_checkpoint(
            args.output_dir,
            model,
            optimizer,
            scaler,
            tokenizer,
            global_step,
            current_epoch,
            config,
            f"error-step-{global_step}",
            scheduler.state_dict() if scheduler is not None else None,
            extra_metadata=checkpoint_metadata,
        )
        raise
    finally:
        if stop_training:
            current_epoch = locals().get("epoch", start_epoch)
            save_checkpoint(
                args.output_dir,
                model,
                optimizer,
                scaler,
                tokenizer,
                global_step,
                current_epoch,
                config,
                f"interrupt-step-{global_step}",
                scheduler.state_dict() if scheduler is not None else None,
                extra_metadata=checkpoint_metadata,
            )

    if nan_triggered_stop:
        print(
            "Entrenamiento detenido por pérdida no finita. "
            "Considera reducir el learning rate, activar clipping de gradiente o "
            "usar --nan_behavior reduce-lr/skip.",
            flush=True,
        )
    elif nan_events > 0:
        print(
            f"Se detectaron {nan_events} microbatches con pérdida no finita; "
            f"acción aplicada: {args.nan_behavior}.",
            flush=True,
        )

    total_elapsed = time.time() - step_start_time
    steps_completed = max(global_step - initial_global_step, 0)
    tokens_trained = steps_completed * tokens_per_step
    avg_tokens_per_sec = (
        tokens_trained / total_elapsed if total_elapsed > 0 else 0.0
    )
    checkpoint_metadata["final_tokens_trained"] = tokens_trained
    checkpoint_metadata["final_tokens_per_sec"] = avg_tokens_per_sec
    checkpoint_metadata["max_vram_gb"] = max_vram_gb
    if not stop_training:
        # Guardar un último checkpoint aún cuando el cierre sea ordenado.
        final_epoch = locals().get("epoch", args.epochs - 1)
        save_checkpoint(
            args.output_dir,
            model,
            optimizer,
            scaler,
            tokenizer,
            global_step,
            final_epoch,
            config,
            f"final-step-{global_step}",
            scheduler.state_dict() if scheduler is not None else None,
            extra_metadata=checkpoint_metadata,
        )
    print(
        f"Pasos completados: {steps_completed} | tokens vistos: {tokens_trained:,} | "
        f"tiempo transcurrido: {format_time(total_elapsed)} | throughput medio: {avg_tokens_per_sec:,.0f} tok/s",
        flush=True,
    )
    if device.type == "cuda":
        print(
            f"Uso máximo de VRAM registrado: {max_vram_gb:.2f} GiB",
            flush=True,
        )

    final_model_path = args.output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    with (args.output_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_preset": args.selected_preset,
                "vocab_size": len(tokenizer),
                "max_seq_length": args.seq_length,
                "embed_size": args.embed_size,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "save_steps": args.save_steps,
                "warmup_steps": args.warmup_steps,
                "gradient_checkpointing": args.gradient_checkpointing,
                "effective_batch_size": effective_batch,
                "tokens_per_step": tokens_per_step,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "micro_batches_per_epoch": micro_batches_per_epoch,
                "precision": args.precision,
                "autocast_dtype": str(args.autocast_dtype) if args.autocast_dtype else None,
                "use_grad_scaler": args.use_grad_scaler,
                "nan_behavior": args.nan_behavior,
                "nan_lr_factor": args.nan_lr_factor,
                "nan_min_lr": args.nan_min_lr,
                "nan_patience": args.nan_patience,
                "nan_events": nan_events,
                "nan_triggered_stop": nan_triggered_stop,
                "steps_completed": steps_completed,
                "last_nan_adjust_step": last_nan_adjust_step,
                "target_hours": args.target_hours,
                "runtime_required_tokens_per_sec": runtime_info.get("required_tokens_per_sec"),
                "runtime_estimated_optimizer_steps": runtime_info.get("estimated_optimizer_steps"),
                "runtime_gpu_free_gb": runtime_info.get("gpu_free_gb"),
                "runtime_gpu_total_gb": runtime_info.get("gpu_total_gb"),
                "max_vram_gb": max_vram_gb,
                "final_tokens_per_sec": avg_tokens_per_sec,
                "dataset_file": str(dataset_file),
                **tokenizer_info,
            },
            f,
            indent=2,
        )
    tokenizer.save_pretrained(args.output_dir)
    print("Entrenamiento completado. Modelo final guardado.", flush=True)


if __name__ == "__main__":
    main()
