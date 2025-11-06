"""Entrenamiento LoRA para Aura en español usando RigoChat-7B-v2."""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets", "aura_personality_es.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models", "lora_aura_es")
BASE_MODEL_ID = "IIC/RigoChat-7b-v2"
MODEL_MAX_LENGTH = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena un adaptador LoRA para Aura en español.")
    parser.add_argument(
        "--dataset-path",
        default=DATASET_PATH,
        help="Ruta al dataset JSONL con campos instruction y output.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directorio donde guardar el adaptador LoRA entrenado.",
    )
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL_ID,
        help="Identificador del modelo base en Hugging Face.",
    )
    return parser.parse_args()


def detect_precision() -> Dict[str, Any]:
    """Detecta si se debe entrenar en 4 bits (QLoRA) o en 16 bits."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024**3)
        print(f"Memoria total de la GPU detectada: {total_gb:.2f} GB")
        if total_gb < 24:
            print("Se utilizará QLoRA (4 bits).")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            return {"quantization_config": bnb_config, "torch_dtype": torch.float16}
        print("Se utilizará entrenamiento en precisión de 16 bits.")
        return {"torch_dtype": torch.float16}
    print("No se detectó GPU, se utilizará CPU en precisión 16 bits (puede ser lento).");
    return {"torch_dtype": torch.float16}


def load_tokenizer(model_id: str) -> AutoTokenizer:
    print("Cargando tokenizador...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MODEL_MAX_LENGTH
    return tokenizer


def load_model(model_id: str, precision_kwargs: Dict[str, Any]) -> AutoModelForCausalLM:
    print("Cargando modelo base...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if torch.cuda.is_available() else None,
        **precision_kwargs,
    )
    model.config.use_cache = False
    return model


def load_training_dataset(dataset_path: str, tokenizer: AutoTokenizer):
    print("Cargando dataset de entrenamiento...")
    dataset = load_dataset("json", data_files=str(dataset_path))

    def format_example(example: Dict[str, str]) -> Dict[str, str]:
        instruction = example["instruction"].strip()
        output = example["output"].strip()
        prompt = (
            "### Instrucción:\n"
            f"{instruction}\n\n"
            "### Respuesta:\n"
            f"{output}\n"
        )
        return {"text": prompt}

    dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

    def tokenize_function(example: Dict[str, str]) -> Dict[str, List[int]]:
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=MODEL_MAX_LENGTH,
            padding="max_length",
        )

    tokenized = dataset["train"].map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch")
    return tokenized


def create_trainer(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, tokenized_dataset, output_dir: str) -> Trainer:
    print("Configurando entrenamiento...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    if hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, peft_config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    return trainer


def main() -> None:
    args = parse_args()
    precision_kwargs = detect_precision()
    tokenizer = load_tokenizer(args.base_model)
    tokenized_dataset = load_training_dataset(args.dataset_path, tokenizer)
    model = load_model(args.base_model, precision_kwargs)
    trainer = create_trainer(model, tokenizer, tokenized_dataset, args.output_dir)

    print("Entrenando adaptador LoRA...")
    trainer.train()

    print("Guardando adaptador entrenado...")
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Entrenamiento finalizado. Adaptador guardado en:", args.output_dir)


if __name__ == "__main__":
    main()
