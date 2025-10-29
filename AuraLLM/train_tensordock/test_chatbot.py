"""Chatbot interactivo para el modelo Aura entrenado en TensorDock."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError("Se requiere transformers para ejecutar el chatbot.") from exc

from AuraLLM.archive.transformer_architecture import Config as ModelConfig, GPT2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chatbot interactivo de Aura")
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Directorio con final_model.pt, model_config.json y el tokenizer",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Tokens máximos por respuesta")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperatura para muestreo")
    parser.add_argument("--top_k", type=int, default=50, help="Filtrado top-k (0 desactiva)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Filtrado top-p (núcleo)")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Penalización de repetición")
    parser.add_argument(
        "--history_turns",
        type=int,
        default=6,
        help="Cantidad de turnos anteriores a mantener en el contexto",
    )
    return parser.parse_args()


def load_artifacts(model_dir: Path) -> tuple[GPT2, ModelConfig, AutoTokenizer]:
    config_path = model_dir / "model_config.json"
    weights_path = model_dir / "final_model.pt"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            "No se encontraron model_config.json o final_model.pt en el directorio indicado"
        )

    with config_path.open("r", encoding="utf-8") as f:
        raw_config = json.load(f)

    model_config = ModelConfig(
        vocab_size=raw_config["vocab_size"],
        max_seq_length=raw_config["max_seq_length"],
        embed_size=raw_config["embed_size"],
        num_layers=raw_config["num_layers"],
        num_heads=raw_config["num_heads"],
        dropout=raw_config.get("dropout", 0.1),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token or "<|pad|>"})

    model = GPT2(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_config, tokenizer


def apply_repetition_penalty(logits: torch.Tensor, generated: List[int], penalty: float) -> torch.Tensor:
    if penalty <= 1.0 or not generated:
        return logits
    logits = logits.clone()
    for token_id in set(generated):
        index = int(token_id)
        value = logits[index]
        if value > 0:
            logits[index] = value / penalty
        else:
            logits[index] = value * penalty
    return logits


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    top_k = max(top_k, 0)
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[-1]
        logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    if 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        cutoff = cumulative_probs > top_p
        if cutoff.any():
            cutoff_index = cutoff.nonzero(as_tuple=False)[0].item()
            sorted_logits[cutoff_index + 1 :] = float("-inf")
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(0, sorted_indices, sorted_logits)
            logits = filtered
    if torch.isinf(logits).all():
        logits = torch.zeros_like(logits)
    return logits


def generate_response(
    model: GPT2,
    tokenizer: AutoTokenizer,
    config: ModelConfig,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> str:
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(device)
    generated: List[int] = input_ids[0].tolist()

    for _ in range(max_new_tokens):
        context = generated[-config.max_seq_length :]
        context_tensor = torch.tensor([context], device=device, dtype=torch.long)
        with torch.no_grad():
            logits = model(context_tensor)
        logits = logits[:, -1, :].squeeze(0)
        logits = apply_repetition_penalty(logits, generated, repetition_penalty)
        logits = logits / max(temperature, 1e-5)
        logits = top_k_top_p_filtering(logits, top_k, top_p)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    new_tokens = generated[len(input_ids[0]) :]
    if not new_tokens:
        return ""
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def run_chat(args: argparse.Namespace) -> None:
    model, config, tokenizer = load_artifacts(args.model_dir)
    history: List[str] = []

    print("Chatbot listo. Escribe 'salir' para terminar.\n", flush=True)
    while True:
        try:
            user_input = input("Usuario: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSalida solicitada.", flush=True)
            break

        if not user_input:
            continue
        if user_input.lower() in {"salir", "exit", "quit"}:
            print("Hasta pronto.", flush=True)
            break

        history.append(f"Usuario: {user_input}")
        history = history[-(2 * args.history_turns) :]
        prompt = "\n".join(history + ["Aura:"])

        response = generate_response(
            model,
            tokenizer,
            config,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            args.repetition_penalty,
        )
        response = response.split("Usuario:")[0].strip()
        if not response:
            response = "Estoy procesando la información, ¿puedes reformular?"
        print(f"Aura: {response}", flush=True)
        history.append(f"Aura: {response}")


def main() -> None:
    args = parse_args()
    run_chat(args)


if __name__ == "__main__":
    main()
