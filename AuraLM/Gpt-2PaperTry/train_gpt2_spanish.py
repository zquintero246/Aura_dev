import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, GPT2Tokenizer, get_linear_schedule_with_warmup


class Config:
    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_length: int = 128,
        embed_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert (
            config.embed_size % config.num_heads == 0
        ), "embed_size debe ser divisible entre num_heads"

        self.num_heads = config.num_heads
        self.head_dim = config.embed_size // config.num_heads

        self.W_q = nn.Linear(config.embed_size, config.embed_size)
        self.W_k = nn.Linear(config.embed_size, config.embed_size)
        self.W_v = nn.Linear(config.embed_size, config.embed_size)
        self.output = nn.Linear(config.embed_size, config.embed_size)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_seq_length, config.max_seq_length))
            .view(1, 1, config.max_seq_length, config.max_seq_length)
            .bool(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_length, embed_dim = x.size()
        Q = self.W_q(x).view(batch, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = attention.masked_fill(
            ~self.mask[:, :, :seq_length, :seq_length], float("-inf")
        )
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        scores = attention @ V
        scores = scores.transpose(1, 2).contiguous().view(batch, seq_length, embed_dim)
        scores = self.output(scores)
        return self.dropout(scores)


class FFN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_size, 4 * config.embed_size)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.embed_size, config.embed_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_size)
        self.attention = MultiHeadSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.embed_size)
        self.mlp = FFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_size)
        self.pos_embed = nn.Embedding(config.max_seq_length, config.embed_size)
        self.dropout = nn.Dropout(config.dropout)
        self.transformers = nn.Sequential(
            *[Transformer(config) for _ in range(config.num_layers)]
        )
        self.norml = nn.LayerNorm(config.embed_size)

    def forward(self, input_token: torch.Tensor) -> torch.Tensor:
        batch, seq_length = input_token.size()
        pos = torch.arange(0, seq_length, dtype=torch.long, device=input_token.device)
        pos = pos.unsqueeze(0)
        x = self.token_embed(input_token) + self.pos_embed(pos)
        x = self.dropout(x)
        x = self.transformers(x)
        x = self.norml(x)
        return x @ self.token_embed.weight.t()


class SpanishCorpus(Dataset):
    def __init__(self, data: torch.Tensor, seq_length: int):
        super().__init__()
        self.text = data
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.text) - self.seq_length

    def __getitem__(self, idx: int):
        x = self.text[idx : idx + self.seq_length]
        y = self.text[idx + 1 : idx + 1 + self.seq_length]
        return x, y


def log_memory_usage(device: torch.device, prefix: str = "") -> None:
    """Imprime el uso de VRAM (si hay GPU) y RAM en el proceso."""

    ram = psutil.virtual_memory()
    ram_msg = (
        f"RAM usada: {ram.used / (1024 ** 3):.2f} GB / "
        f"{ram.total / (1024 ** 3):.2f} GB"
    )
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        vram_msg = f"VRAM usada: {allocated:.1f} MB / {total:.1f} MB"
        print(f"{prefix}{vram_msg} | {ram_msg}", flush=True)
    else:
        print(f"{prefix}{ram_msg}", flush=True)


@dataclass
class DistributedState:
    """Mantiene metadatos mínimos del entorno de entrenamiento actual."""

    is_distributed: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = torch.device("cpu")
    backend: Optional[str] = None


def unwrap_distributed_model(model: nn.Module) -> nn.Module:
    """En modo mono-GPU no existe un envoltorio adicional, se devuelve tal cual."""

    return model


def seq_len_from_model(model: nn.Module) -> int:
    """Obtiene la longitud máxima de contexto soportada por el modelo actual."""

    if hasattr(model, "config") and getattr(model.config, "n_positions", None):
        return int(model.config.n_positions)
    if hasattr(model, "pos_embed"):
        return int(model.pos_embed.num_embeddings)
    return 128


def setup(args: argparse.Namespace) -> DistributedState:
    """Inicializa un entorno de entrenamiento local mono-proceso."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Se fuerza el uso del primer dispositivo disponible para evitar
        # inicializaciones accidentales sobre CPU cuando sí hay GPU.
        torch.cuda.set_device(0)
        gpu_name = torch.cuda.get_device_name(device)
        print(f"[Local] Entrenamiento en GPU: {gpu_name} ({device})", flush=True)
    else:
        print("[Local] Entrenamiento en CPU (no se detectó GPU CUDA).", flush=True)

    return DistributedState(
        is_distributed=False,
        rank=0,
        world_size=1,
        local_rank=0,
        device=device,
        backend=None,
    )


def cleanup() -> None:
    """No se requiere limpieza adicional en modo mono-proceso."""

    return None


def sample(
    model: nn.Module,
    device: torch.device,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    length: int = 50,
    temperature: float = 1.0,
    seq_len: int = 128,
    using_hf_model: bool = False,
) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    if using_hf_model and hasattr(model, "generate"):
        max_context = min(seq_len, getattr(model.config, "n_positions", seq_len))
        total_max = min(input_ids.size(1) + length, max_context)
        total_max = max(total_max, input_ids.size(1) + 1)
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_length=total_max,
                temperature=max(temperature, 1e-6),
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(generated[0], skip_special_tokens=True)

    tokens = input_ids
    for _ in range(length):
        tokens_ = tokens[:, -seq_len:]
        with torch.no_grad():
            scores = model(tokens_)
        next_token_scores = scores[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(next_token_scores, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokenizer.decode(tokens[0], skip_special_tokens=True)


def reset_parameters(module: nn.Module) -> None:
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    using_hf_model: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    progress = tqdm(loader, desc="Validación", leave=False, disable=False)
    with torch.no_grad():
        for x, y in progress:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                if using_hf_model:
                    outputs = model(input_ids=x, labels=y)
                    loss = outputs.loss
                else:
                    scores = model(x)
                    loss = F.cross_entropy(
                        scores.view(-1, scores.size(-1)), y.view(-1)
                    )
            total_loss += loss.item()
            total_steps += 1
            progress.set_postfix(loss=loss.item())

    return total_loss / max(1, total_steps)


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
    model_to_save = unwrap_distributed_model(model)
    state = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if scaler.is_enabled():
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, checkpoint_path)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    rank: int,
    distributed: bool,
    train_sampler: Optional[Sampler],
    val_sampler: Optional[Sampler],
    epochs: int = 5,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    checkpoint_dir: Path = Path("./checkpoints"),
    checkpoint_freq: int = 1,
    sample_prompt: str = "Hola, ¿cómo",
    sample_length: int = 50,
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
    log_interval: int = 100,
    memory_report_interval: int = 100,
    eval_frequency: int = 1,
    using_hf_model: bool = False,
) -> nn.Module:
    del rank, distributed, train_sampler, val_sampler  # Compatibilidad de firma.

    use_amp = device.type == "cuda"
    dataset_size = len(train_loader.dataset)
    print(
        f"[Local] Entrenando con {dataset_size} muestras totales en modo mono-GPU.",
        flush=True,
    )

    gradient_accumulation_steps = max(1, gradient_accumulation_steps)
    updates_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_updates = max(1, epochs * updates_per_epoch)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )
    warmup_steps = max(1, int(total_updates * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates
    )
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        if distributed and train_sampler is not None:
            # Cada proceso baraja un subconjunto distinto del dataset.
            train_sampler.set_epoch(epoch)

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            leave=False,
            disable=False,
        )

        for step, (x, y) in enumerate(progress, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                if using_hf_model:
                    outputs = model(input_ids=x, labels=y)
                    loss = outputs.loss
                else:
                    scores = model(x)
                    loss = F.cross_entropy(
                        scores.view(-1, scores.size(-1)), y.view(-1)
                    )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Se detectó un valor de pérdida no finito. Reduce el learning rate "
                    "o verifica los datos de entrada."
                )

            loss_value = loss.item()
            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            should_step = step % gradient_accumulation_steps == 0 or step == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            total_loss += loss_value

            if log_interval and step % log_interval == 0:
                progress.set_postfix(loss=loss_value)
                print(
                    f"[Epoch {epoch} | Step {step}] Loss actual: {loss_value:.4f}",
                    flush=True,
                )

            if memory_report_interval and step % memory_report_interval == 0:
                log_memory_usage(device, prefix=f"[Epoch {epoch} | Step {step}] ")

        avg_train_loss = total_loss / max(1, len(train_loader))

        val_loss = None
        if val_loader is not None and (epoch % max(1, eval_frequency) == 0):
            val_loss = evaluate(
                model,
                val_loader,
                device,
                use_amp,
                using_hf_model=using_hf_model,
            )

        summary = f"Epoch {epoch}/{epochs} | Loss entrenamiento: {avg_train_loss:.4f}"
        if val_loss is not None:
            summary += f" | Loss validación: {val_loss:.4f}"
        print(summary, flush=True)

        if is_main_process:
            print(
                f"Epoch {epoch}/{epochs} | Loss entrenamiento: {avg_train_loss:.4f} | "
                f"Loss validación: {val_loss:.4f}"
            )

        if is_main_process and checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
            save_checkpoint(checkpoint_dir, epoch, model, optimizer, scheduler, scaler)

        if sample_prompt:
            try:
                base_model = unwrap_distributed_model(model)
                generated = sample(
                    base_model,
                    device,
                    tokenizer,
                    prompt=sample_prompt,
                    length=sample_length,
                    seq_len=seq_len_from_model(base_model),
                    using_hf_model=using_hf_model,
                )
                print("=== Ejemplo de texto generado ===")
                print(generated)
                print("=================================")
            except Exception as exc:  # pragma: no cover - logging únicamente
                print("Error al generar texto de ejemplo:", exc)

        log_memory_usage(device, prefix=f"[Epoch {epoch} fin] ")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("[Local] Entrenamiento completado.", flush=True)

    return unwrap_distributed_model(model)


def load_dataset(tensor_path: Path) -> torch.Tensor:
    if not tensor_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {tensor_path}")
    tensor = torch.load(tensor_path, map_location='cpu', mmap=True)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"El archivo {tensor_path} no contiene un tensor de PyTorch")
    return tensor.long()


def tokenize_text_stream(
    dataset_iter: Iterable[dict],
    tokenizer: GPT2Tokenizer,
    eos_id: int,
    max_samples: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> torch.Tensor:
    token_ids = []
    tokens_target = max_tokens if max_tokens is not None else float("inf")
    sample_target = max_samples if max_samples is not None else float("inf")

    progress = tqdm(dataset_iter, desc="Tokenizando corpus", total=max_samples, leave=False)

    for idx, sample in enumerate(progress, start=1):
        if idx - 1 >= sample_target:
            break
        text = sample.get("text", "")
        if not text:
            continue
        text = text.strip()
        if not text:
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        ids.append(eos_id)
        token_ids.extend(ids)

        if len(token_ids) >= tokens_target:
            token_ids = token_ids[: int(tokens_target)]
            break

        if max_samples is not None:
            progress.set_postfix(samples=idx, tokens=len(token_ids))
        else:
            progress.set_postfix(tokens=len(token_ids))

    progress.close()

    if len(token_ids) < 2:
        raise RuntimeError("No se obtuvieron tokens suficientes del corpus indicado.")

    return torch.tensor(token_ids, dtype=torch.long)


def prepare_hf_dataset(
    data_dir: Path,
    tokenizer: GPT2Tokenizer,
    seq_len: int,
    val_ratio: float,
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    max_samples: Optional[int],
    max_tokens: Optional[int],
    force: bool = False,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - se informa al usuario
        raise ImportError(
            "Se requiere el paquete 'datasets' para descargar el corpus seleccionado. "
            "Instálalo con `pip install datasets`"
        ) from exc

    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train_ids.pt"
    val_path = data_dir / "val_ids.pt"

    if not force and train_path.exists() and val_path.exists():
        return

    dataset_id: str
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower == "oscar":
        dataset_id = "oscar-corpus/OSCAR-2201"
        dataset_kwargs = {"trust_remote_code": False}
    elif dataset_name_lower == "wikipedia":
        # El dataset "wikipedia" original basado en scripts fue retirado en
        # versiones recientes de `datasets`. El reemplazo oficial es
        # "wikimedia/wikipedia", el cual funciona sin scripts externos.
        dataset_id = "wikimedia/wikipedia"
        dataset_kwargs = {"trust_remote_code": False}
    else:
        raise ValueError(
            "dataset-name debe ser 'oscar' o 'wikipedia'."
        )

    print(f"Descargando y tokenizando {dataset_id} ({dataset_config}, {dataset_split})…")
    try:
        dataset = load_dataset(
            dataset_id,
            dataset_config,
            split=dataset_split,
            **dataset_kwargs,
        )
    except RuntimeError as exc:
        if dataset_name_lower == "wikipedia" and "no longer supported" in str(exc).lower():
            raise RuntimeError(
                "El dataset 'wikipedia' clásico basado en scripts ya no está "
                "disponible en versiones recientes de `datasets`. Usa el "
                "identificador 'wikimedia/wikipedia' (configuramos esto "
                "automáticamente) y asegúrate de tener la versión >= 2.15 del "
                "paquete `datasets`."
            ) from exc
        raise

    eos_id = tokenizer.eos_token_id
    tokens = tokenize_text_stream(
        dataset,
        tokenizer=tokenizer,
        eos_id=eos_id,
        max_samples=max_samples,
        max_tokens=max_tokens,
    )

    total_tokens = tokens.numel()
    if total_tokens <= seq_len:
        raise RuntimeError(
            "El número de tokens es insuficiente. Ajusta los parámetros de descarga."
        )

    train_min = seq_len + 1
    val_min = seq_len + 1
    max_train = total_tokens - val_min

    if max_train < train_min:
        raise RuntimeError(
            "No hay suficientes tokens para satisfacer los tamaños mínimos de train/val."
        )

    desired_train = int(total_tokens * (1 - val_ratio))
    cutoff = max(train_min, min(desired_train, max_train))

    train_ids = tokens[:cutoff]
    val_ids = tokens[cutoff:]

    torch.save(train_ids, train_path)
    torch.save(val_ids, val_path)

    print(
        {
            "tokens_total": total_tokens,
            "train_tokens": train_ids.numel(),
            "val_tokens": val_ids.numel(),
            "val_ratio": val_ratio,
            "dataset": dataset_id,
            "dataset_config": dataset_config,
            "dataset_split": dataset_split,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento GPT-2 español")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Descargar y tokenizar el dataset antes de entrenar",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument(
        "--dataset-name",
        default="wikipedia",
        choices=["oscar", "wikipedia"],
        help="Nombre del dataset de Hugging Face a utilizar",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Configuración del dataset (idioma/version)",
    )
    parser.add_argument(
        "--dataset-split",
        default=None,
        help="Split del dataset a descargar",
    )
    parser.add_argument("--dataset-max-samples", type=int, default=5000)
    parser.add_argument("--dataset-max-tokens", type=int, default=300_000)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignorar caches existentes y regenerar el dataset",
    )
    parser.add_argument(
        "--sample-prompt",
        default="Hola, ¿cómo",
        help="Texto inicial para las muestras generadas",
    )
    parser.add_argument("--sample-length", type=int, default=40)
    parser.add_argument(
        "--hf-model",
        default="distilgpt2",
        help="Modelo de Hugging Face a cargar en modo liviano",
    )
    parser.add_argument(
        "--use-custom-model",
        action="store_true",
        help="Utiliza la arquitectura GPT-2 ligera definida en este archivo",
    )
    parser.add_argument("--embed-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--model-dropout", type=float, default=0.1)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Número de pasos para acumular gradientes antes de aplicar un update",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Valor máximo para el clipping de gradientes",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=200,
        help="Frecuencia (en pasos) para imprimir la pérdida intermedia",
    )
    parser.add_argument(
        "--memory-report-interval",
        type=int,
        default=100,
        help="Cada cuántos pasos reportar uso de memoria. 0 desactiva el reporte",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=1,
        help="Frecuencia de evaluación sobre el conjunto de validación (en epochs)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Número de workers para los DataLoader (usar valores bajos ahorra RAM)",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="distilgpt2",
        help="Nombre del tokenizer a cargar desde Hugging Face",
    )
    parser.add_argument(
        "--disable-sampling",
        action="store_true",
        help="No generar muestras de texto al final de cada epoch",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Activa gradient checkpointing en modelos de Hugging Face",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        dest="cudnn_benchmark",
        action="store_true",
        help="Activa torch.backends.cudnn.benchmark para optimizar convoluciones",
    )
    parser.add_argument(
        "--no-cudnn-benchmark",
        dest="cudnn_benchmark",
        action="store_false",
        help="Desactiva torch.backends.cudnn.benchmark",
    )
    parser.set_defaults(cudnn_benchmark=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dist_state = setup(args)
    device = dist_state.device

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "Data"

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_prefix_space = True
    tokenizer.pad_token = tokenizer.eos_token

    dataset_dir = data_dir / args.dataset_name.lower()

    dataset_config = args.dataset_config
    dataset_split = args.dataset_split
    if dataset_config is None:
        dataset_config = "es" if args.dataset_name == "oscar" else "20231101.es"
    if dataset_split is None:
        dataset_split = "train[:0.5%]" if args.dataset_name == "oscar" else "train[:1%]"

    if args.prepare or args.force_download or not (dataset_dir / "train_ids.pt").exists():
        if is_main_process:
            prepare_hf_dataset(
                dataset_dir,
                tokenizer,
                seq_len=args.seq_len,
                val_ratio=args.val_ratio,
                dataset_name=args.dataset_name,
                dataset_config=dataset_config,
                dataset_split=dataset_split,
                max_samples=args.dataset_max_samples,
                max_tokens=args.dataset_max_tokens,
                force=args.force_download,
            )
        if distributed:
            dist.barrier()

    train_ids = load_dataset(dataset_dir / "train_ids.pt")
    val_ids = load_dataset(dataset_dir / "val_ids.pt")

    seq_len = args.seq_len
    if args.use_custom_model:
        if args.embed_size % args.num_heads != 0:
            raise ValueError("embed-size debe ser divisible entre num-heads")
        config = Config(
            vocab_size=tokenizer.vocab_size,
            max_seq_length=seq_len,
            embed_size=args.embed_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.model_dropout,
        )
        model = GPT2(config)
        using_hf_model = False
    else:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )
        model.resize_token_embeddings(tokenizer.vocab_size)
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.eos_token_id
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        using_hf_model = True

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros del modelo: {total_params / 1e6:.2f} M", flush=True)

    train_dataset = SpanishCorpus(train_ids, seq_length=seq_len)
    val_dataset = SpanishCorpus(val_ids, seq_length=seq_len)

    num_workers = max(0, min(args.num_workers, os.cpu_count() or 1))
    pin_memory = device.type == "cuda"
    persistent_workers = pin_memory and num_workers > 0
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_sampler: Optional[DistributedSampler]
    val_sampler: Optional[DistributedSampler]
    if distributed:
        # El DistributedSampler divide el dataset entre los procesos globales para que
        # cada uno procese un fragmento distinto sin solaparse con el resto. torchrun
        # proporciona rank/world_size, lo que permite que el sampler calcule la porción
        # correspondiente en cada nodo.
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
        gpu_name = torch.cuda.get_device_name(device)
        print(f"Usando GPU {gpu_name} (dispositivo {device})", flush=True)
    else:
        torch.backends.cudnn.benchmark = False
        print("Usando CPU", flush=True)

    log_memory_usage(device, prefix="[Inicio] ")

    sample_prompt = "" if args.disable_sampling else args.sample_prompt
    trained_model = train(
        model,
        train_loader,
        val_loader,
        tokenizer,
        device,
        rank=0,
        distributed=False,
        train_sampler=None,
        val_sampler=None,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        checkpoint_dir=base_dir / "checkpoints",
        checkpoint_freq=args.checkpoint_freq,
        sample_prompt=sample_prompt,
        sample_length=args.sample_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        memory_report_interval=args.memory_report_interval,
        eval_frequency=args.eval_frequency,
        using_hf_model=using_hf_model,
    )

    model_path = base_dir / "gpt2_spanish.pth"
    torch.save(trained_model.state_dict(), model_path)
    print(f"Modelo guardado en {model_path}", flush=True)


if __name__ == "__main__":
    # Ejecutar directamente:
    #   python train_gpt2_spanish.py --prepare
    try:
        main()
    finally:
        cleanup()

