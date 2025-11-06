"""Descarga el modelo base RigoChat-7B-v2 en formato GGUF."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:  # pragma: no cover - dependencia opcional en runtime
    raise SystemExit(
        "huggingface_hub no está instalado. Ejecuta 'pip install huggingface_hub'"
    ) from exc


DEFAULT_REPO_ID = "IIC/RigoChat-7b-v2-GGUF"
DEFAULT_FILENAME = "RigoChat-7B-v2.Q4_K_M.gguf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Descarga un archivo GGUF de Hugging Face y lo guarda como aura.gguf en la carpeta models/."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Repositorio en Hugging Face desde el que descargar el archivo GGUF.",
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help="Nombre del archivo GGUF dentro del repositorio.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "models", "aura.gguf"),
        help="Ruta de salida para guardar el archivo descargado.",
    )
    return parser.parse_args()


def ensure_output_dir(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    ensure_output_dir(output_path)

    print(
        "Descargando archivo GGUF desde Hugging Face...\n"
        f"  Repositorio: {args.repo_id}\n"
        f"  Archivo: {args.filename}"
    )
    downloaded_path = hf_hub_download(repo_id=args.repo_id, filename=args.filename)

    print(f"Copiando archivo a {output_path}...")
    data = Path(downloaded_path).read_bytes()
    output_path.write_bytes(data)

    print("Descarga completada. Archivo disponible en:", output_path)


if __name__ == "__main__":
    main()

