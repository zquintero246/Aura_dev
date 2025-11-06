"""Script interactivo para probar el adaptador LoRA o el modelo fusionado de Aura."""
from __future__ import annotations

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = "### Instrucción:\n{instruction}\n\n### Respuesta:\n"
DEFAULT_ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "models", "lora_aura_es")
DEFAULT_MERGED_DIR = os.path.join(os.path.dirname(__file__), "models", "aura_es_merged")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prueba rápida del modelo Aura entrenado con LoRA.")
    parser.add_argument("prompt", nargs="?", help="Instrucción inicial para generar una respuesta.")
    parser.add_argument("--base-model", default="IIC/RigoChat-7b-v2", help="Identificador del modelo base en Hugging Face.")
    parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER_DIR,
        help="Ruta al adaptador LoRA entrenado (por defecto models/lora_aura_es).",
    )
    parser.add_argument(
        "--merged",
        default=None,
        help="Ruta al modelo ya fusionado. Si se especifica, no se carga el adaptador LoRA.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Número máximo de tokens a generar.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperatura para sampling.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p para sampling.")
    return parser.parse_args()


def load_tokenizer(path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_merged(path: str, device: torch.device):
    tokenizer = load_tokenizer(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    return tokenizer, model


def load_model(args: argparse.Namespace, device: torch.device):
    if args.merged:
        print("Cargando modelo fusionado...")
        tokenizer, model = _load_merged(args.merged, device)
    else:
        adapter_dir = args.adapter
        adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")

        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(
                f"No se encontró el directorio del adaptador en '{adapter_dir}'. Ejecuta el entrenamiento antes de probar."
            )

        if not os.path.isfile(adapter_config_path):
            merged_candidate = DEFAULT_MERGED_DIR if os.path.isdir(DEFAULT_MERGED_DIR) else adapter_dir
            print(
                "No se encontró adapter_config.json en el adaptador. "
                f"Se intentará cargar '{merged_candidate}' como modelo fusionado."
            )
            tokenizer, model = _load_merged(merged_candidate, device)
        else:
            print("Cargando modelo base y adaptador LoRA...")
            tokenizer = load_tokenizer(args.base_model)
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device_map="auto" if device.type == "cuda" else None,
            )
            model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


def build_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def extract_response(tokenizer: AutoTokenizer, generated_ids) -> str:
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if "### Respuesta:\n" in text:
        return text.split("### Respuesta:\n", 1)[1].strip()
    return text.strip()


def chat_loop(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo detectado: {device}")
    tokenizer, model = load_model(args, device)

    def generate_once(instruction: str) -> str:
        prompt = build_prompt(instruction)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        return extract_response(tokenizer, outputs[0])

    if args.prompt:
        print("\nRespuesta de Aura:\n")
        print(generate_once(args.prompt))
        return

    print("Ingresa instrucciones en español. Escribe 'salir' para terminar.\n")
    while True:
        try:
            instruction = input("Usuario: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nHasta luego.")
            break
        if instruction.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego.")
            break
        if not instruction:
            continue
        response = generate_once(instruction)
        print(f"Aura: {response}\n")


def main() -> None:
    args = parse_args()
    chat_loop(args)


if __name__ == "__main__":
    main()
