import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

if os.name == "nt" and "TORCH_DISTRIBUTED_USE_LIBUV" not in os.environ:
    # En Windows, algunas distribuciones de PyTorch no incluyen soporte para libuv.
    # torchrun intenta usar libuv por defecto para el rendezvous TCP y falla con
    # "use_libuv was requested but PyTorch was build without libuv support".
    # Forzamos el uso del backend basado en sockets tradicionales dentro del
    # proceso de entrenamiento. Aun así, es recomendable establecer esta
    # variable en la shell antes de invocar torchrun para evitar el fallo
    # durante el rendezvous inicial.
    os.environ["TORCH_DISTRIBUTED_USE_LIBUV"] = "0"

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup


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


@dataclass
class DistributedState:
    """Metadatos sobre el contexto distribuido activo."""

    is_distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: Optional[str]


def unwrap_distributed_model(model: nn.Module) -> nn.Module:
    """Devuelve el módulo real, incluso si está envuelto por DDP."""

    return model.module if isinstance(model, DistributedDataParallel) else model


def setup(args: argparse.Namespace) -> DistributedState:
    """Configura el contexto distribuido (o single-process) y devuelve su estado."""

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    cli_world_size = args.world_size if args.world_size is not None else env_world_size
    should_distribute = (
        args.distributed
        or cli_world_size > 1
        or env_world_size > 1
        or (args.rank is not None and args.rank >= 0)
    )

    if not should_distribute:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            "[Rank 0] Ejecutando en modo single-process. "
            f"Dispositivo: {device}. WORLD_SIZE=1",
            flush=True,
        )
        return DistributedState(
            is_distributed=False,
            rank=0,
            world_size=1,
            local_rank=0,
            device=device,
            backend=None,
        )

    master_addr = args.master_addr or os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = args.master_port or os.environ.get("MASTER_PORT", "29500")
    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)

    world_size = cli_world_size if cli_world_size is not None else env_world_size
    if world_size < 1:
        raise ValueError("WORLD_SIZE debe ser mayor o igual a 1")
    os.environ["WORLD_SIZE"] = str(world_size)

    rank = args.rank if args.rank is not None and args.rank >= 0 else int(os.environ.get("RANK", "0"))
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK debe estar en el rango [0, {world_size - 1}]")
    os.environ["RANK"] = str(rank)

    local_rank = (
        args.local_rank
        if args.local_rank is not None and args.local_rank >= 0
        else int(os.environ.get("LOCAL_RANK", "0"))
    )
    os.environ["LOCAL_RANK"] = str(local_rank)

    preferred_backend = "nccl" if torch.cuda.is_available() else "gloo"
    backend = preferred_backend
    init_error: Optional[RuntimeError] = None
    try:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    except RuntimeError as exc:
        init_error = exc
        if preferred_backend == "nccl":
            backend = "gloo"
            print(
                f"[Rank {rank}] Falló la inicialización NCCL ({exc}). "
                "Reintentando con backend 'gloo'.",
                flush=True,
            )
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                rank=rank,
                world_size=world_size,
            )
        else:
            raise

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_index = local_rank if 0 <= local_rank < device_count else 0
        if local_rank >= device_count:
            print(
                f"[Rank {rank}] Advertencia: LOCAL_RANK {local_rank} excede las GPUs disponibles "
                f"({device_count}). Se usará la GPU 0.",
                flush=True,
            )
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
    else:
        device = torch.device("cpu")

    debug_msg = (
        f"[Rank {rank}] Proceso inicializado | backend={backend} | "
        f"world_size={world_size} | local_rank={local_rank} | "
        f"MASTER={master_addr}:{master_port}"
    )
    if init_error is not None and backend == "gloo":
        debug_msg += " | fallback=gloo"
    print(debug_msg, flush=True)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(
            f"[Rank {rank}] Usando GPU '{gpu_name}' en el índice {device.index}.",
            flush=True,
        )
    else:
        print(f"[Rank {rank}] Usando CPU.", flush=True)

    return DistributedState(
        is_distributed=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        backend=backend,
    )


def cleanup() -> None:
    """Libera los recursos del proceso distribuido si están inicializados."""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def sample(
    model: GPT2,
    device: torch.device,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    length: int = 50,
    temperature: float = 1.0,
    seq_len: int = 128,
) -> str:
    model.eval()
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
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
    distributed: bool,
    rank: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    progress = tqdm(loader, desc="Validación", leave=False, disable=rank != 0)
    with torch.no_grad():
        for x, y in progress:
            x = x.to(device)
            y = y.to(device)
            with autocast(enabled=use_amp):
                scores = model(x)
                loss = F.cross_entropy(
                    scores.view(-1, scores.size(-1)), y.view(-1)
                )
            total_loss += loss.item()
            total_steps += 1
            if rank == 0:
                progress.set_postfix(loss=loss.item())

    if distributed:
        stats = torch.tensor([total_loss, total_steps], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_steps = stats.tolist()

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
    train_sampler: Optional[DistributedSampler],
    val_sampler: Optional[DistributedSampler],
    epochs: int = 5,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    checkpoint_dir: Path = Path("./checkpoints"),
    checkpoint_freq: int = 1,
    sample_prompt: str = "Hola, ¿cómo",
    sample_length: int = 50,
) -> nn.Module:
    use_amp = device.type == "cuda"
    is_main_process = rank == 0
    dataset_size = len(train_loader.dataset)
    if distributed:
        print(
            f"[Rank {rank}] Preparando entrenamiento distribuido con {dataset_size} muestras locales.",
            flush=True,
        )
        print(
            f"[Rank {rank}] Sincronizando procesos antes de iniciar el entrenamiento…",
            flush=True,
        )
        dist.barrier()
        print(
            f"[Rank {rank}] Sincronización inicial completada. Comenzando epochs.",
            flush=True,
        )
    else:
        print(
            f"[Rank {rank}] Entrenamiento en modo single-process con {dataset_size} muestras.",
            flush=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(train_loader))
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler(enabled=use_amp)

    epoch = 0
    while epoch < epochs:
        epoch += 1
        model.train()
        total_loss = 0.0
        total_steps_epoch = 0
        nan_detected = False

        if distributed and train_sampler is not None:
            # Cada proceso baraja un subconjunto distinto del dataset.
            train_sampler.set_epoch(epoch)

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs} - Entrenamiento (rank {rank})",
            leave=False,
            disable=not is_main_process,
        )

        for step, (x, y) in enumerate(progress, start=1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                scores = model(x)
                loss = F.cross_entropy(
                    scores.view(-1, scores.size(-1)), y.view(-1)
                )

            if torch.isnan(loss) or torch.isinf(loss):
                nan_detected = True
                break

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_value = loss.item()
            total_loss += loss_value
            total_steps_epoch += 1
            if is_main_process:
                progress.set_postfix(loss=loss_value)

        if distributed:
            nan_tensor = torch.tensor([1 if nan_detected else 0], device=device)
            dist.all_reduce(nan_tensor, op=dist.ReduceOp.SUM)
            nan_detected = nan_tensor.item() > 0

            stats = torch.tensor([total_loss, total_steps_epoch], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, total_steps_epoch = stats.tolist()

        if nan_detected:
            if is_main_process:
                print(
                    "Se detectó una pérdida NaN. Reiniciando entrenamiento desde cero."
                )
            base_model = unwrap_distributed_model(model)
            base_model.apply(reset_parameters)

            if distributed:
                # Sincroniza los nuevos pesos entre todos los procesos para evitar divergencias.
                state_dict = base_model.state_dict() if is_main_process else None
                obj_list = [state_dict]
                dist.broadcast_object_list(obj_list, src=0)
                if not is_main_process:
                    unwrap_distributed_model(model).load_state_dict(obj_list[0])
                dist.barrier()

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            scaler = GradScaler(enabled=use_amp)
            epoch = 0
            continue

        avg_train_loss = total_loss / max(1, total_steps_epoch)

        if distributed and val_sampler is not None:
            val_sampler.set_epoch(epoch)

        val_loss = evaluate(
            model,
            val_loader,
            device,
            use_amp,
            distributed,
            rank,
        )

        if is_main_process:
            print(
                f"Epoch {epoch}/{epochs} | Loss entrenamiento: {avg_train_loss:.4f} | "
                f"Loss validación: {val_loss:.4f}"
            )

        if is_main_process and checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
            save_checkpoint(checkpoint_dir, epoch, model, optimizer, scheduler, scaler)

        if is_main_process:
            try:
                base_model = unwrap_distributed_model(model)
                generated = sample(
                    base_model,
                    device,
                    tokenizer,
                    prompt=sample_prompt,
                    length=sample_length,
                    seq_len=base_model.pos_embed.num_embeddings,
                )
                print("=== Ejemplo de texto generado ===")
                print(generated)
                print("=================================")
            except Exception as exc:  # pragma: no cover - logging únicamente
                print("Error al generar texto de ejemplo:", exc)

    if distributed:
        print(
            f"[Rank {rank}] Esperando sincronización final tras completar los epochs…",
            flush=True,
        )
        dist.barrier()
        print(
            f"[Rank {rank}] Todos los procesos completaron el entrenamiento.",
            flush=True,
        )
    else:
        print("[Rank 0] Entrenamiento single-process completado.", flush=True)

    # Devolvemos el modelo base (sin el envoltorio DDP) para facilitar guardados o
    # evaluaciones posteriores en un único proceso.
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--checkpoint-freq", type=int, default=1)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Descargar y tokenizar el dataset antes de entrenar",
    )
    parser.add_argument("--val-ratio", type=float, default=0.01)
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
    parser.add_argument("--dataset-max-samples", type=int, default=None)
    parser.add_argument("--dataset-max-tokens", type=int, default=None)
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
    parser.add_argument("--sample-length", type=int, default=50)
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Fuerza la inicialización de torch.distributed aunque WORLD_SIZE=1",
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default=None,
        help="Dirección IP o hostname del nodo maestro (MASTER_ADDR)",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="Puerto TCP del nodo maestro (MASTER_PORT)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Rank global del proceso actual (RANK)",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Número total de procesos en todos los nodos (WORLD_SIZE)",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="Rank local dentro del nodo (LOCAL_RANK, usado por torchrun)",
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
    parser.set_defaults(cudnn_benchmark=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dist_state = setup(args)
    rank = dist_state.rank
    world_size = dist_state.world_size
    distributed = dist_state.is_distributed
    device = dist_state.device
    is_main_process = rank == 0

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "Data"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_prefix_space = True
    tokenizer.pad_token = tokenizer.eos_token

    dataset_dir = data_dir / args.dataset_name.lower()

    dataset_config = args.dataset_config
    dataset_split = args.dataset_split
    if dataset_config is None:
        dataset_config = "es" if args.dataset_name == "oscar" else "20231101.es"
    if dataset_split is None:
        dataset_split = "train[:1%]" if args.dataset_name == "oscar" else "train"

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
    config = Config(
        vocab_size=tokenizer.vocab_size,
        max_seq_length=seq_len,
        embed_size=768,
        num_layers=12,
        num_heads=12,
        dropout=0.1,
    )

    train_dataset = SpanishCorpus(train_ids, seq_length=config.max_seq_length)
    val_dataset = SpanishCorpus(val_ids, seq_length=config.max_seq_length)

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
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=val_sampler is None,
        sampler=val_sampler,
        drop_last=False,
    )

    if device.type == "cuda":
        cudnn_flag = True if args.cudnn_benchmark is None else bool(args.cudnn_benchmark)
        torch.backends.cudnn.benchmark = cudnn_flag
    else:
        torch.backends.cudnn.benchmark = False
    print(
        f"[Rank {rank}] cudnn.benchmark={'ON' if torch.backends.cudnn.benchmark else 'OFF'}",
        flush=True,
    )

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(
            f"[Rank {rank}] usando GPU {gpu_name} (dispositivo {device})",
            flush=True,
        )
    else:
        print(f"[Rank {rank}] usando CPU", flush=True)

    model = GPT2(config).to(device)

    if distributed:
        # En modo distribuido cada proceso conserva únicamente su porción del modelo en la GPU
        # asociada (local_rank). Al envolver el modelo con DDP, PyTorch se encarga de
        # sincronizar automáticamente los gradientes mediante operaciones all-reduce al final
        # de cada backward(). Esto garantiza que todos los nodos/núcleos vean el mismo estado de
        # entrenamiento sin que tengamos que escribir llamadas manuales a dist.all_reduce.
        # Cuando el script se ejecuta en modo single-GPU o CPU, se omite este envoltorio y el
        # flujo permanece idéntico al entrenamiento tradicional.
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    trained_model = train(
        model,
        train_loader,
        val_loader,
        tokenizer,
        device,
        rank,
        distributed,
        train_sampler,
        val_sampler,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        checkpoint_dir=base_dir / "checkpoints",
        checkpoint_freq=args.checkpoint_freq,
        sample_prompt=args.sample_prompt,
        sample_length=args.sample_length,
    )

    if is_main_process:
        model_path = base_dir / "gpt2_spanish.pth"
        torch.save(trained_model.state_dict(), model_path)
        print(f"Modelo guardado en {model_path}", flush=True)


if __name__ == "__main__":
    # Ejemplo (WSL) de lanzamiento distribuido con torchrun para 2 nodos y 1 GPU por nodo:
    # node0:
    #   MASTER_ADDR=192.168.1.10 MASTER_PORT=12355 WORLD_SIZE=2 RANK=0 LOCAL_RANK=0 \
    #   torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
    #       --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    #       train_gpt2_spanish.py --distributed --batch-size=8 --seq-len=128
    # node1:
    #   MASTER_ADDR=192.168.1.10 MASTER_PORT=12355 WORLD_SIZE=2 RANK=1 LOCAL_RANK=0 \
    #   torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
    #       --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    #       train_gpt2_spanish.py --distributed --batch-size=8 --seq-len=128
    try:
        main()
    finally:
        cleanup()

