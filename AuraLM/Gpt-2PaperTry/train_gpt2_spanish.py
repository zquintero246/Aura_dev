import argparse
import math
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
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
    model: GPT2,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    progress = tqdm(loader, desc="Validación", leave=False)
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
            progress.set_postfix(loss=loss.item())
    return total_loss / max(1, len(loader))


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: GPT2,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if scaler.is_enabled():
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, checkpoint_path)


def train(
    model: GPT2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    epochs: int = 5,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    checkpoint_dir: Path = Path("./checkpoints"),
    checkpoint_freq: int = 1,
    sample_prompt: str = "Hola, ¿cómo",
    sample_length: int = 50,
) -> GPT2:
    use_amp = device.type == "cuda"
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
        nan_detected = False

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs} - Entrenamiento",
            leave=False,
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
            progress.set_postfix(loss=loss_value)

        if nan_detected:
            print("Se detectó una pérdida NaN. Reiniciando entrenamiento desde cero.")
            model.apply(reset_parameters)
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

        avg_train_loss = total_loss / max(1, len(train_loader))
        val_loss = evaluate(model, val_loader, device, use_amp)

        print(
            f"Epoch {epoch}/{epochs} | Loss entrenamiento: {avg_train_loss:.4f} | "
            f"Loss validación: {val_loss:.4f}"
        )

        if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
            save_checkpoint(checkpoint_dir, epoch, model, optimizer, scheduler, scaler)

        try:
            generated = sample(
                model,
                device,
                tokenizer,
                prompt=sample_prompt,
                length=sample_length,
                seq_len=model.pos_embed.num_embeddings,
            )
            print("=== Ejemplo de texto generado ===")
            print(generated)
            print("=================================")
        except Exception as exc:  # pragma: no cover - logging únicamente
            print("Error al generar texto de ejemplo:", exc)

    return model


def load_dataset(tensor_path: Path) -> torch.Tensor:
    if not tensor_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {tensor_path}")
    tensor = torch.load(tensor_path)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo en uso:", device)

    model = GPT2(config).to(device)

    model = train(
        model,
        train_loader,
        val_loader,
        tokenizer,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        checkpoint_dir=base_dir / "checkpoints",
        checkpoint_freq=args.checkpoint_freq,
        sample_prompt=args.sample_prompt,
        sample_length=args.sample_length,
    )

    model_path = base_dir / "gpt2_spanish.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en {model_path}")


if __name__ == "__main__":
    main()