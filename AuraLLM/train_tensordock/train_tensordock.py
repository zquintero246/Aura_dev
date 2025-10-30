"""Entrenamiento de Aura en TensorDock usando una sola GPU A100."""
from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


if __package__ is None or __package__ == "":  # ejecución directa (python AuraLLM/...)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase, get_linear_schedule_with_warmup
except ImportError as exc:  # pragma: no cover - dependencia externa
    raise ImportError(
        "Se requiere transformers. Instala con `pip install transformers`."
    ) from exc

from AuraLLM.train.train_aura import (  # type: ignore
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
        "aura-72h-extended": {
            "embed_size": 2048,
            "num_layers": 28,
            "num_heads": 16,
            "seq_length": 2048,
        },
        "aura-72h-max": {
            "embed_size": 2560,
            "num_layers": 32,
            "num_heads": 20,
            "seq_length": 2048,
        },
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento desde cero de Aura en TensorDock")
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=Path("/datasets/spanish_corpus"),
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
        "--log_interval",
        type=int,
        default=50,
        help="Actualizar barra de progreso cada N pasos acumulados",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Activa checkpointing para capas Transformer (reduce VRAM, aumenta cómputo)",
    )
    parser.add_argument(
        "--checkpoint_segments",
        type=int,
        default=4,
        help="Segmentos para checkpoint_sequential cuando está activo el gradient checkpointing",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Norma máxima para clipping de gradiente (None desactiva)",
    )

    args = parser.parse_args()

    if args.hf_dataset_preset:
        preset = HF_DATASET_PRESETS[args.hf_dataset_preset]
        for key, value in preset.items():
            setattr(args, key, value)

    args.output_dir = args.output_dir.expanduser().resolve()

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
        args.save_steps = 500 if preset in {"aura-72h-max", "aura-72h-extended"} else 1000

    args.batch_size = max(1, int(args.batch_size))
    args.gradient_accumulation_steps = max(1, int(args.gradient_accumulation_steps))
    args.learning_rate = float(args.learning_rate)
    args.save_steps = max(1, int(args.save_steps))

    if args.grad_clip is not None and args.grad_clip <= 0:
        args.grad_clip = None
    if args.gradient_checkpointing:
        args.checkpoint_segments = max(1, int(args.checkpoint_segments))


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
        tokenizer = train_tokenizer_from_corpus(
            dataset_path,
            args.tokenizer_vocab_size,
            args.tokenizer_min_frequency,
            tokenizer_dir,
        )
        return tokenizer, {"tokenizer_source": "auto", "tokenizer_dir": str(tokenizer_dir)}

    tokenizer = load_tokenizer(args.tokenizer_name_or_path)
    saved_dir = args.output_dir / "tokenizer"
    tokenizer.save_pretrained(saved_dir)
    return tokenizer, {
        "tokenizer_source": args.tokenizer_name_or_path,
        "tokenizer_dir": str(saved_dir),
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
    all_ids = []
    for chunk in corpus:
        ids = tokenizer.encode(chunk, add_special_tokens=False)
        if not ids:
            continue
        all_ids.extend(ids)
    if not all_ids:
        raise ValueError("El corpus tokenizado quedó vacío")
    return torch.tensor(all_ids, dtype=torch.long)


def prepare_datasets(
    token_ids: torch.Tensor,
    seq_length: int,
    validation_split: float,
) -> Tuple[SpanishCorpus, Optional[SpanishCorpus]]:
    if len(token_ids) <= seq_length:
        raise ValueError("El corpus es demasiado pequeño para el seq_length especificado")

    if validation_split <= 0 or validation_split >= 1:
        train_dataset = SpanishCorpus(token_ids, seq_length)
        return train_dataset, None

    val_tokens = int(len(token_ids) * validation_split)
    val_tokens = max(seq_length + 1, val_tokens)
    train_tokens = len(token_ids) - val_tokens
    if train_tokens <= seq_length:
        raise ValueError("validation_split demasiado grande para el tamaño del corpus")

    train_ids = token_ids[:train_tokens].clone()
    val_ids = token_ids[train_tokens:].clone()
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
            "scaler_state_dict": scaler.state_dict(),
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
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
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
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
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

    token_ids = tokenize_corpus(tokenizer, read_corpus(dataset_file))
    train_dataset, val_dataset = prepare_datasets(token_ids, args.seq_length, args.validation_split)

    train_tokens = len(train_dataset.text)
    val_tokens = len(val_dataset.text) if val_dataset is not None else 0
    total_tokens = train_tokens + val_tokens
    print(
        f"Tokens disponibles -> entrenamiento: {train_tokens:,} | validación: {val_tokens:,} | total: {total_tokens:,}",
        flush=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )

    micro_batches_per_epoch = len(train_loader)
    if micro_batches_per_epoch == 0:
        raise ValueError("No hay batches disponibles. Reduce batch_size o revisa el corpus.")

    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / args.gradient_accumulation_steps
    )
    total_optimizer_steps = optimizer_steps_per_epoch * args.epochs
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

    model, config = create_model(len(tokenizer), args, device)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model, args.checkpoint_segments)
        print(
            f"Gradient checkpointing activado con {args.checkpoint_segments} segmentos",
            flush=True,
        )

    model.train()
    num_params = sum(p.numel() for p in model.parameters())
    print(
        f"Parámetros del modelo: {num_params/1e9:.2f}B ({num_params:,} parámetros)",
        flush=True,
    )

    optimizer = create_optimizer(model, args.learning_rate, args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

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

    if args.resume_from is not None:
        print(f"Reanudando desde {args.resume_from}", flush=True)
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        global_step = checkpoint.get("global_step", 0)
        start_epoch = checkpoint.get("epoch", 0)
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        extra = checkpoint.get("extra_metadata") or {}
        if extra:
            print(f"Metadatos restaurados: {extra}", flush=True)
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
        **tokenizer_info,
    }

    def handle_interrupt(signum, frame):  # pragma: no cover - señal externa
        nonlocal stop_training
        stop_training = True
        print("Interrupción recibida, se guardará checkpoint al finalizar el paso en curso", flush=True)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    step_start_time = time.time()

    try:
        for epoch in range(start_epoch, args.epochs):
            if stop_training:
                break
            epoch_losses = []
            progress_bar = tqdm(train_loader, desc=f"Época {epoch + 1}/{args.epochs}", leave=False)
            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                if stop_training:
                    break
                inputs = inputs.to(device)
                labels = labels.to(device)

                amp_context = (
                    autocast(device_type="cuda", dtype=torch.float16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with amp_context:
                    logits = model(inputs)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                    )
                loss_value = loss.item()
                loss = loss / args.gradient_accumulation_steps

                scaler.scale(loss).backward()
                micro_step += 1
                epoch_losses.append(loss_value)

                if micro_step == args.gradient_accumulation_steps:
                    if args.grad_clip is not None:
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

                    if args.log_interval > 0 and global_step % args.log_interval == 0:
                        current_lr = optimizer.param_groups[0]["lr"]
                        progress_bar.set_postfix(
                            loss=f"{loss_value:.4f}",
                            lr=f"{current_lr:.2e}",
                            eta=eta,
                            toks=f"{avg_tokens_per_sec:,.0f}",
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

            train_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
            if not stop_training and micro_step > 0:
                if args.grad_clip is not None:
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
                val_loss = evaluate(model, val_loader, device)

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

    total_elapsed = time.time() - step_start_time
    steps_completed = max(global_step - initial_global_step, 0)
    tokens_trained = steps_completed * tokens_per_step
    avg_tokens_per_sec = (
        tokens_trained / total_elapsed if total_elapsed > 0 else 0.0
    )
    print(
        f"Pasos completados: {steps_completed} | tokens vistos: {tokens_trained:,} | "
        f"tiempo transcurrido: {format_time(total_elapsed)} | throughput medio: {avg_tokens_per_sec:,.0f} tok/s",
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
                "steps_completed": steps_completed,
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
