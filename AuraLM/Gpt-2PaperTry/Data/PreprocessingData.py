from pathlib import Path
import re
import torch
from transformers import GPT2Tokenizer

# === Config ===
folder = Path(r"/AuraLM/Gpt-2PaperTry/Data/spanish_corpus")
pattern = "spanishText_*"      # coincide con todos tus shards
block_size = 1024              # o 512/2048 según tu modelo
val_ratio = 0.01               # 1% validación

# === 1) Listar archivos ordenados por rango numérico ===
files = sorted(
    folder.glob(pattern),
    key=lambda p: tuple(map(int, re.findall(r"\d+", p.name)))  # ordena por [inicio, fin]
)

assert files, f"No se encontraron archivos con patrón {pattern} en {folder}"

# === 2) Cargar tokenizer una sola vez ===
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token                    # GPT-2 no trae pad; usamos eos como pad
eos_id = tokenizer.eos_token_id

# === 3) Leer, tokenizar y concatenar ===
all_ids = []  # OJO: para corpus enorme, considera ir flush a disco en shards o usar memmap
for fp in files:
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        # Si el archivo es "una línea por documento", esto va bien.
        # Si trae párrafos largos, igual sirve: añadimos EOS entre docs.
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line, add_special_tokens=False)
            ids.append(eos_id)               # separador entre documentos
            all_ids.extend(ids)

# === 4) Recortar al múltiplo de block_size y reshaped ===
n_full = len(all_ids) // block_size
if n_full == 0:
    raise RuntimeError("Muy pocos tokens para formar al menos un bloque.")

all_ids = all_ids[: n_full * block_size]
data = torch.tensor(all_ids, dtype=torch.long).view(n_full, block_size)

# === 5) Split train/val y guardar ===
cut = int((1.0 - val_ratio) * n_full)
train_data = data[:cut]
val_data   = data[cut:]

out_dir = folder.parent  # guarda al lado de la carpeta 'spanish_corpus'
torch.save(train_data, out_dir / f"train_gpt2_es_{block_size}.pt")
torch.save(val_data,   out_dir / f"val_gpt2_es_{block_size}.pt")

print({
    "num_files": len(files),
    "tokens_total": len(all_ids),
    "blocks_total": n_full,
    "train_blocks": train_data.size(0),
    "val_blocks": val_data.size(0),
    "block_size": block_size,
})
