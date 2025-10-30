from __future__ import annotations

import argparse
import math
import os
import socket
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, GPT2Tokenizer, get_linear_schedule_with_warmup


DEFAULT_SEQ_LEN = 128
DEFAULT_MODEL_DROPOUT = 0.1

CUSTOM_MODEL_PRESETS: Dict[str, Dict[str, int]] = {
    "aura-small": {"embed_size": 768, "num_layers": 12, "num_heads": 12, "seq_len": 1024},
    "aura-medium": {"embed_size": 1024, "num_layers": 24, "num_heads": 16, "seq_len": 1024},
    "aura-large": {"embed_size": 1280, "num_layers": 36, "num_heads": 20, "seq_len": 1024},
}


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


def log_memory_usage(
    device: torch.device,
    prefix: str = "",
    state: Optional[DistributedState] = None,
) -> None:
    """Imprime el uso de VRAM (si hay GPU) y RAM en el proceso."""

    ram = psutil.virtual_memory()
    ram_msg = (
        f"RAM usada: {ram.used / (1024 ** 3):.2f} GB / "
        f"{ram.total / (1024 ** 3):.2f} GB"
    )
    process_prefix = state.formatted_prefix() if state else ""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        vram_msg = f"VRAM usada: {allocated:.1f} MB / {total:.1f} MB"
        print(f"{process_prefix}{prefix}{vram_msg} | {ram_msg}", flush=True)
    else:
        print(f"{process_prefix}{prefix}{ram_msg}", flush=True)


@dataclass
class DistributedState:
    """Mantiene metadatos del entorno de entrenamiento distribuido."""

    is_distributed: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    node_rank: int = 0
    device: torch.device = torch.device("cpu")
    backend: Optional[str] = None
    node_name: str = socket.gethostname()

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def formatted_prefix(self) -> str:
        device_repr = str(self.device)
        return (
            f"[Rank {self.rank}/{self.world_size} | Node {self.node_name} | "
            f"Device {device_repr}] "
        )


def unwrap_distributed_model(model: nn.Module) -> nn.Module:
    """Devuelve el modelo subyacente si está envuelto en DDP."""

    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def seq_len_from_model(model: nn.Module) -> int:
    """Obtiene la longitud máxima de contexto soportada por el modelo actual."""

    if hasattr(model, "config") and getattr(model.config, "n_positions", None):
        return int(model.config.n_positions)
    if hasattr(model, "pos_embed"):
        return int(model.pos_embed.num_embeddings)
    return 128


def _select_backend(requested: str) -> str:
    if requested == "auto":
        return "nccl" if torch.cuda.is_available() else "gloo"
    return requested


def setup(args: argparse.Namespace) -> DistributedState:
    """Inicializa el entorno distribuido usando variables de entorno o CLI."""

    os.environ.setdefault("TORCH_DISTRIBUTED_USE_LIBUV", "0")

    # Permite configurar parámetros de red vía CLI o variables de entorno.
    # Estos valores son necesarios cuando torchrun se ejecuta en más de un nodo.
    # Si se modifica el tamaño del clúster (nnodes o GPUs por nodo) basta con
    # actualizar WORLD_SIZE/NODE_RANK antes de invocar el script.
    if getattr(args, "master_addr", None):
        os.environ["MASTER_ADDR"] = str(args.master_addr)
    if getattr(args, "master_port", None):
        os.environ["MASTER_PORT"] = str(args.master_port)
    if getattr(args, "world_size", None):
        os.environ["WORLD_SIZE"] = str(args.world_size)
    if getattr(args, "rank", None) is not None:
        os.environ["RANK"] = str(args.rank)
    if getattr(args, "node_rank", None) is not None:
        os.environ["NODE_RANK"] = str(args.node_rank)
        # Cuando se lanza el script manualmente sin torchrun se utiliza NODE_RANK
        # como RANK global para mantener compatibilidad.
        os.environ.setdefault("RANK", str(args.node_rank))

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    rank_env = int(os.environ.get("RANK", os.environ.get("NODE_RANK", "0")))
    local_rank_env = int(os.environ.get("LOCAL_RANK", "0"))
    node_rank_env = int(os.environ.get("NODE_RANK", "0"))

    distributed = world_size_env > 1
    backend = None

    if distributed:
        requested_backend = _select_backend(getattr(args, "backend", "auto"))
        try:
            dist.init_process_group(
                backend=requested_backend,
                init_method="env://",
                rank=rank_env,
                world_size=world_size_env,
            )
            backend = requested_backend
        except RuntimeError as exc:
            if requested_backend == "nccl":
                # NCCL puede fallar en algunos entornos Windows/WSL.
                print(
                    f"[Rank {rank_env}] NCCL no disponible ({exc}). Se intentará con 'gloo'.",
                    flush=True,
                )
                dist.init_process_group(
                    backend="gloo",
                    init_method="env://",
                    rank=rank_env,
                    world_size=world_size_env,
                )
                backend = "gloo"
            else:
                raise

        # Una vez inicializado el proceso distribuido se obtienen los valores reales.
        rank_env = dist.get_rank()
        world_size_env = dist.get_world_size()
    else:
        backend = None

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_index = local_rank_env if distributed else 0
        if device_index >= device_count:
            device_index = device_index % max(1, device_count)
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(device)
        print(
            f"[Setup] GPU activa: {gpu_name} (cuda:{device_index}). benchmark=True",
            flush=True,
        )
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False
        print("[Setup] Entrenamiento en CPU (sin CUDA disponible).", flush=True)

    node_name = socket.gethostname()
    state = DistributedState(
        is_distributed=distributed,
        rank=rank_env,
        world_size=world_size_env,
        local_rank=local_rank_env,
        node_rank=node_rank_env,
        device=device,
        backend=backend,
        node_name=node_name,
    )

    prefix = state.formatted_prefix()
    mode = "Distribuido" if distributed else "Local"
    print(
        f"{prefix}Modo {mode}. Backend={backend or 'N/A'}. Sincronizando...",
        flush=True,
    )
    if distributed:
        dist.barrier()
    print(f"{prefix}Sincronización inicial completada.", flush=True)

    return state


def cleanup(state: Optional[DistributedState] = None) -> None:
    """Cierra el grupo de procesos distribuido si está activo."""

    if dist.is_available() and dist.is_initialized():
        if state and state.is_distributed:
            dist.barrier()
        dist.destroy_process_group()


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
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
    using_hf_model: bool,
    distributed: bool,
    world_size: int,
    is_main_process: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    progress = tqdm(
        loader,
        desc="Validación",
        leave=False,
        disable=not is_main_process,
    )
    with torch.no_grad():
        for x, y in progress:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            context = (
                autocast(device_type=device.type, dtype=amp_dtype)
                if amp_enabled
                else nullcontext()
            )
            with context:
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
            if is_main_process:
                progress.set_postfix(loss=loss.item())

    if distributed:
        stats = torch.tensor([total_loss, float(total_steps)], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = stats[0].item()
        total_steps = int(stats[1].item())

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
    val_loader: Optional[DataLoader],
    tokenizer: GPT2Tokenizer,
    dist_state: DistributedState,
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
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
    log_interval: int = 100,
    memory_report_interval: int = 100,
    eval_frequency: int = 1,
    using_hf_model: bool = False,
    amp_enabled: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    scaler_enabled: bool = False,
    writer: Optional[SummaryWriter] = None,
) -> nn.Module:
    device = dist_state.device
    rank = dist_state.rank
    world_size = dist_state.world_size
    distributed = dist_state.is_distributed
    is_main_process = dist_state.is_main_process
    prefix = dist_state.formatted_prefix()

    dataset_size = len(train_loader.dataset)
    batches_per_epoch = len(train_loader)
    print(
        f"{prefix}Entrenando con {dataset_size} muestras totales. "
        f"Batches locales/epoch: {batches_per_epoch}.",
        flush=True,
    )

    gradient_accumulation_steps = max(1, gradient_accumulation_steps)
    updates_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation_steps)
    total_updates = max(1, epochs * updates_per_epoch)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )
    warmup_steps = max(1, int(total_updates * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates
    )
    scaler = GradScaler(enabled=scaler_enabled)

    psutil.cpu_percent(interval=None)  # Inicializa el cálculo de porcentaje de CPU.
    global_update = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        steps_this_epoch = 0

        if distributed and train_sampler is not None:
            # Cada epoch necesita una semilla diferente para evitar que los
            # procesos lean los mismos lotes. Al modificar el número de nodos
            # o GPUs hay que asegurarse de que WORLD_SIZE coincida para que
            # el muestreador reparta el dataset de forma uniforme.
            train_sampler.set_epoch(epoch)

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            leave=False,
            disable=not is_main_process,
        )

        for step, (x, y) in enumerate(progress, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            context = (
                autocast(device_type=device.type, dtype=amp_dtype)
                if amp_enabled
                else nullcontext()
            )
            with context:
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
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = (
                step % gradient_accumulation_steps == 0 or step == batches_per_epoch
            )
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_update += 1

                if writer and is_main_process:
                    cpu_usage = psutil.cpu_percent(interval=None)
                    ram_usage = psutil.virtual_memory().percent
                    writer.add_scalar("Loss/train_step", loss_value, global_update)
                    writer.add_scalar("System/CPU_percent", cpu_usage, global_update)
                    writer.add_scalar("System/RAM_percent", ram_usage, global_update)
                    if device.type == "cuda":
                        vram = torch.cuda.memory_allocated(device) / (1024 ** 2)
                        writer.add_scalar("System/GPU_VRAM_MB", vram, global_update)

            total_loss += loss_value
            steps_this_epoch += 1

            if log_interval and step % log_interval == 0:
                progress.set_postfix(loss=loss_value)
                sync_status = "OK" if distributed else "Local"
                print(
                    f"{prefix}Epoch {epoch} | Step {step} | Loss: {loss_value:.4f} | "
                    f"Sync={sync_status}",
                    flush=True,
                )

            if memory_report_interval and step % memory_report_interval == 0:
                log_memory_usage(
                    device,
                    prefix=f"Epoch {epoch} | Step {step} | ",
                    state=dist_state,
                )

        loss_tensor = torch.tensor(
            [total_loss, float(steps_this_epoch)], device=device
        )
        if distributed:
            # all_reduce garantiza que todos los nodos compartan la misma
            # estadística de pérdida. Esto confirma que los gradientes fueron
            # sincronizados correctamente por DDP.
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = (loss_tensor[0] / loss_tensor[1]).item()

        val_loss = None
        if (
            val_loader is not None
            and (epoch % max(1, eval_frequency) == 0)
            and len(val_loader) > 0
        ):
            val_loss = evaluate(
                model,
                val_loader,
                device,
                amp_enabled,
                amp_dtype,
                using_hf_model=using_hf_model,
                distributed=distributed,
                world_size=world_size,
                is_main_process=is_main_process,
            )

        epoch_time = time.time() - epoch_start
        summary = (
            f"{prefix}Epoch {epoch}/{epochs} | Loss entrenamiento: {avg_train_loss:.4f}"
        )
        if val_loss is not None:
            summary += f" | Loss validación: {val_loss:.4f}"
        summary += f" | Duración epoch: {epoch_time:.1f}s"
        print(summary, flush=True)

        if writer and is_main_process:
            writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
            if val_loss is not None:
                writer.add_scalar("Loss/val_epoch", val_loss, epoch)
            writer.add_scalar("Time/epoch_seconds", epoch_time, epoch)

        if is_main_process and checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
            save_checkpoint(checkpoint_dir, epoch, model, optimizer, scheduler, scaler)

        if is_main_process and sample_prompt:
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
                print("=== Ejemplo de texto generado (rank 0) ===")
                print(generated)
                print("=========================================")
            except Exception as exc:  # pragma: no cover - logging únicamente
                print("Error al generar texto de ejemplo:", exc, flush=True)

        log_memory_usage(device, prefix="Fin epoch | ", state=dist_state)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if distributed:
        dist.barrier()
    print(f"{prefix}Entrenamiento completado. Gradientes sincronizados.", flush=True)

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
            "No hay suficientes tokens para satisfacer los tamaños mínimos de train_DDP/val."
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
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
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
        "--precision",
        choices=["fp32", "fp16", "bf16", "auto"],
        default="fp32",
        help=(
            "Precisión numérica para el modelo. 'fp16' y 'bf16' requieren CUDA. "
            "El modo 'auto' intenta usar fp16 en GPU y fp32 en CPU."
        ),
    )
    parser.add_argument(
        "--use-custom-model",
        action="store_true",
        help="Utiliza la arquitectura GPT-2 ligera definida en este archivo",
    )
    parser.add_argument(
        "--custom-preset",
        choices=sorted(CUSTOM_MODEL_PRESETS.keys()),
        default=None,
        help=(
            "Selecciona un preset para la arquitectura personalizada (gpt2-small, "
            "gpt2-medium, gpt2-large). Sobrescribe embed-size, num-heads, num-layers "
            "y, si no se modificó manualmente, la longitud de secuencia."
        ),
    )
    parser.add_argument("--embed-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--model-dropout", type=float, default=DEFAULT_MODEL_DROPOUT)
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
    parser.add_argument(
        "--backend",
        choices=["auto", "nccl", "gloo"],
        default="auto",
        help="Backend de torch.distributed (auto selecciona NCCL si hay GPU).",
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default=None,
        help="Dirección IP o hostname del nodo maestro (MASTER_ADDR).",
    )
    parser.add_argument(
        "--master-port",
        type=str,
        default=None,
        help="Puerto TCP para la inicialización distribuida (MASTER_PORT).",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Número total de procesos (nnodes * nproc_per_node).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Rank global del proceso actual (se usa si no se define vía torchrun).",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=None,
        help="Índice del nodo actual dentro del clúster (NODE_RANK).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="Directorio donde se guardarán los logs de TensorBoard.",
    )
    parser.set_defaults(cudnn_benchmark=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.custom_preset and not args.use_custom_model:
        if os.environ.get("RANK") in (None, "0"):
            print(
                "[Aviso] --custom-preset requiere la arquitectura personalizada. "
                "Activando --use-custom-model automáticamente.",
                flush=True,
            )
        args.use_custom_model = True

    if args.use_custom_model and args.custom_preset:
        preset = CUSTOM_MODEL_PRESETS[args.custom_preset]
        args.embed_size = preset["embed_size"]
        args.num_layers = preset["num_layers"]
        args.num_heads = preset["num_heads"]
        if args.model_dropout == DEFAULT_MODEL_DROPOUT and "dropout" in preset:
            args.model_dropout = float(preset["dropout"])
        if args.seq_len == DEFAULT_SEQ_LEN and "seq_len" in preset:
            args.seq_len = preset["seq_len"]

    dist_state = setup(args)
    device = dist_state.device
    distributed = dist_state.is_distributed
    is_main_process = dist_state.is_main_process

    requested_precision = args.precision.lower()
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if requested_precision == "auto":
        target_dtype = torch.float16 if device.type == "cuda" else torch.float32
    else:
        target_dtype = dtype_map.get(requested_precision, torch.float32)

    if target_dtype in (torch.float16, torch.bfloat16) and device.type != "cuda":
        if is_main_process:
            print(
                f"{dist_state.formatted_prefix()}Precisión '{args.precision}' requiere CUDA. Se usará fp32.",
                flush=True,
            )
        target_dtype = torch.float32

    if target_dtype == torch.bfloat16 and device.type == "cuda":
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if not bf16_supported:
            if is_main_process:
                print(
                    f"{dist_state.formatted_prefix()}La GPU no soporta bf16. Se usará fp32.",
                    flush=True,
                )
            target_dtype = torch.float32

    amp_enabled = device.type == "cuda" and target_dtype in (
        torch.float16,
        torch.bfloat16,
    )
    scaler_enabled = device.type == "cuda" and target_dtype == torch.float16

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
        dataset_split = "train_DDP[:0.5%]" if args.dataset_name == "oscar" else "train_DDP[:1%]"

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
        if is_main_process:
            preset_name = args.custom_preset or "manual"
            print(
                f"{dist_state.formatted_prefix()}Config personalizada '{preset_name}': "
                f"d_model={args.embed_size}, layers={args.num_layers}, heads={args.num_heads}, "
                f"seq_len={seq_len}, dropout={args.model_dropout}",
                flush=True,
            )
        config = Config(
            vocab_size=tokenizer.vocab_size,
            max_seq_length=seq_len,
            embed_size=args.embed_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.model_dropout,
        )
        model = GPT2(config)
        model = model.to(dtype=target_dtype)
        using_hf_model = False
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            low_cpu_mem_usage=True,
            torch_dtype=target_dtype,
        )
        model.resize_token_embeddings(tokenizer.vocab_size)
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.eos_token_id
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        using_hf_model = True

    model = model.to(device=device, dtype=target_dtype)

    if distributed:
        # El envoltorio DDP replica el modelo en todos los procesos. Al cambiar
        # el número de GPUs por nodo, torchrun se encargará de ajustar LOCAL_RANK
        # para cada proceso, por lo que no es necesario modificar este bloque.
        ddp_kwargs = {"broadcast_buffers": False, "gradient_as_bucket_view": True}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [device.index]
            ddp_kwargs["output_device"] = device.index
        model = DistributedDataParallel(model, **ddp_kwargs)

    total_params = sum(p.numel() for p in unwrap_distributed_model(model).parameters())
    if is_main_process:
        print(
            f"{dist_state.formatted_prefix()}Parámetros del modelo: {total_params / 1e6:.2f} M",
            flush=True,
        )
        precision_label = {
            torch.float32: "fp32",
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
        }.get(target_dtype, str(target_dtype))
        print(
            f"{dist_state.formatted_prefix()}Precisión activa: {precision_label} | "
            f"Autocast={'ON' if amp_enabled else 'OFF'} | "
            f"GradScaler={'ON' if scaler_enabled else 'OFF'}",
            flush=True,
        )

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
        # Cada sampler recibe world_size y rank. Ajustar estos valores cuando
        # cambie la cantidad de nodos o GPUs para asegurar que cada proceso
        # procese una porción exclusiva del dataset.
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist_state.world_size,
            rank=dist_state.rank,
            shuffle=True,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist_state.world_size,
            rank=dist_state.rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        shuffle=False,
        **loader_kwargs,
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
        gpu_name = torch.cuda.get_device_name(device)
        print(
            f"{dist_state.formatted_prefix()}Usando GPU {gpu_name} (dispositivo {device}).",
            flush=True,
        )
    else:
        torch.backends.cudnn.benchmark = False
        print(f"{dist_state.formatted_prefix()}Usando CPU", flush=True)

    log_memory_usage(device, prefix="Inicio | ", state=dist_state)

    sample_prompt = "" if args.disable_sampling else args.sample_prompt
    writer: Optional[SummaryWriter] = None
    if is_main_process:
        writer = SummaryWriter(log_dir=args.log_dir)

    trained_model = train(
        model,
        train_loader,
        val_loader,
        tokenizer,
        dist_state,
        train_sampler,
        val_sampler,
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
        amp_enabled=amp_enabled,
        amp_dtype=target_dtype if amp_enabled else None,
        scaler_enabled=scaler_enabled,
        writer=writer,
    )

    if writer is not None:
        writer.flush()
        writer.close()

    if is_main_process:
        model_path = base_dir / "gpt2_spanish.pth"
        torch.save(trained_model.state_dict(), model_path)
        print(f"{dist_state.formatted_prefix()}Modelo guardado en {model_path}", flush=True)

    return dist_state


if __name__ == "__main__":
    # Ejecutar directamente:
    #   python train_aura.py --prepare
    _state: Optional[DistributedState] = None
    try:
        _state = main()
    finally:
        cleanup(_state)

