"""Fusiona el adaptador LoRA de Aura con el modelo base."""
from __future__ import annotations

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "IIC/RigoChat-7b-v2"
LORA_DIR = os.path.join(os.path.dirname(__file__), "models", "lora_aura_es")
MERGED_DIR = os.path.join(os.path.dirname(__file__), "models", "aura_es_merged")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fusiona un adaptador LoRA con el modelo base.")
    parser.add_argument("--base-model", default=BASE_MODEL_ID, help="Modelo base a utilizar.")
    parser.add_argument("--lora-path", default=LORA_DIR, help="Ruta del adaptador LoRA entrenado.")
    parser.add_argument("--output-dir", default=MERGED_DIR, help="Directorio de salida del modelo fusionado.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Cargando modelo base para fusión...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print("Cargando adaptador LoRA...")
    model = PeftModel.from_pretrained(model, args.lora_path)

    print("Fusionando pesos...")
    merged_model = model.merge_and_unload()

    print("Guardando modelo fusionado en formato safetensors...")
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)

    print("Guardando tokenizador asociado...")
    tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
    tokenizer.save_pretrained(args.output_dir)

    print("Fusión completada. Modelo guardado en:", args.output_dir)


if __name__ == "__main__":
    main()
