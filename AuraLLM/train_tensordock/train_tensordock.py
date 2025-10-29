"""Entrenamiento de Aura en TensorDock usando una sola GPU A100."""
from __future__ import annotations

import argparse
import json
import math
import signal
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
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
    GPT2,
    SpanishCorpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento desde cero de Aura en TensorDock")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Ruta del corpus (.txt o .jsonl)")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directorio donde guardar checkpoints y modelos")
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True, help="Nombre o ruta local del tokenizer en español")
    parser.add_argument("--epochs", type=int, default=1, help="Número de épocas completas")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamaño de batch real (antes de acumulación)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Pasos a acumular antes de optimizar")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Tasa de aprendizaje inicial")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay para AdamW")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Pasos de warmup para el scheduler lineal")
    parser.add_argument("--seq_length", type=int, default=1024, help="Longitud de contexto (tokens)")
    parser.add_argument("--embed_size", type=int, default=768, help="Dimensión de embeddings")
    parser.add_argument("--num_layers", type=int, default=12, help="Número de capas Transformer")
    parser.add_argument("--num_heads", type=int, default=12, help="Cabezas de atención por capa")
    parser.add_argument("--dropout", type=float, default=0.1, help="Probabilidad de dropout")
    parser.add_argument("--validation_split", type=float, default=0.01, help="Proporción para validación (0 desactiva)")
    parser.add_argument("--save_steps", type=int, default=1000, help="Guardar checkpoint cada N pasos de optimización")
    parser.add_argument("--resume_from", type=Path, default=None, help="Ruta a un checkpoint previo")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    parser.add_argument("--num_workers", type=int, default=4, help="Workers de DataLoader")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Actualizar barra de progreso cada N pasos acumulados",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(path_or_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(path_or_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token or "<|pad|>"})
    tokenizer.model_max_length = int(1e12)  # sin truncamiento implícito
    return tokenizer


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
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        print(
            f"Usando GPU {properties.name} con {properties.total_memory / (1024**3):.1f} GB", flush=True
        )
    else:
        print("Advertencia: no se detectó GPU, el entrenamiento será muy lento", flush=True)

    tokenizer = load_tokenizer(args.tokenizer_name_or_path)
    token_ids = tokenize_corpus(tokenizer, read_corpus(args.dataset_path))
    train_dataset, val_dataset = prepare_datasets(token_ids, args.seq_length, args.validation_split)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
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
        )

    model, config = create_model(len(tokenizer), args, device)
    model.train()
    optimizer = create_optimizer(model, args.learning_rate, args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    total_optimizer_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs
    scheduler = None
    if args.warmup_steps > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_optimizer_steps,
        )

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
        model.train()

    initial_global_step = global_step

    if len(train_loader) == 0:
        raise ValueError("No hay batches disponibles. Reduce batch_size o revisa el corpus.")

    stop_training = False

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

                    if args.log_interval > 0 and global_step % args.log_interval == 0:
                        current_lr = optimizer.param_groups[0]["lr"]
                        progress_bar.set_postfix(
                            loss=f"{loss_value:.4f}",
                            lr=f"{current_lr:.2e}",
                            eta=eta,
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
                        )

                    if stop_training:
                        break

            train_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
            if not stop_training and micro_step > 0:
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
            )

    final_model_path = args.output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    with (args.output_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab_size": len(tokenizer),
                "max_seq_length": args.seq_length,
                "embed_size": args.embed_size,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
            },
            f,
            indent=2,
        )
    tokenizer.save_pretrained(args.output_dir)
    print("Entrenamiento completado. Modelo final guardado.", flush=True)


if __name__ == "__main__":
    main()
