from transformers import AutoTokenizer
import os
import numpy as np

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
tok.save_pretrained("tokenizer_gpt2_es")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

path = r"C:\Users\Zabdiel Julian\Downloads\IngenieriaSoftwareII\Niggahush_dev\NiggahushLM\test\data\corpus_poems.txt"
with open(path, "r", encoding="utf-8") as f:
    corpus = f.read()

enc = tok(corpus, add_special_tokens=True)
input_ids = enc["input_ids"]

print("Ejemplo de IDs:", input_ids[:40])
print("Total tokens:", len(input_ids))


# === 1) Split por tokens: 95% / 5% ===
ids = np.array(input_ids, dtype=np.int32)
N = len(ids)
split = int(N * 0.95)

train_ids = ids[:split]
valid_ids = ids[split:]

print(f"Total tokens: {N:,} | train: {len(train_ids):,} | valid: {len(valid_ids):,}")

# === 2) Construir bloques (x,y) sin solapamiento ===
# y es x desplazado en +1 (next-token prediction)
def build_xy(ids: np.ndarray, block_size: int):
    # necesitamos pares (x: ids[:-1], y: ids[1:]) y que ambos sean múltiplos de block_size
    usable = (len(ids) - 1) // block_size * block_size  # descarta el sobrante al final
    x = ids[:usable].reshape(-1, block_size)
    y = ids[1:usable+1].reshape(-1, block_size)
    return x, y

block_size = 256

train_x, train_y = build_xy(train_ids, block_size)
valid_x, valid_y = build_xy(valid_ids, block_size)

print(f"Bloques train: {len(train_x):,} | Bloques valid: {len(valid_x):,} | block_size: {block_size}")

# === 3) Sanity checks rápidos ===
# a) dimensiones correctas
assert train_x.shape == train_y.shape and valid_x.shape == valid_y.shape
assert train_x.shape[1] == block_size and valid_x.shape[1] == block_size

# b) desplazamiento correcto: y es x corrido 1
#   verificamos en el primer bloque unas cuantas posiciones
sample_x = train_x[0][:20].tolist()
sample_y = train_y[0][:20].tolist()
print("x[0][:20] ids:", sample_x)
print("y[0][:20] ids:", sample_y)

# c) decodifica 30 tokens para ver que y empieza "un token después" que x
print("x[0][:30] text:", tok.decode(train_x[0][:30].tolist()))
print("y[0][:30] text:", tok.decode(train_y[0][:30].tolist()))

# === 4) Guardar en disco para cargar rápido en el DataLoader ===
os.makedirs("data_blocks", exist_ok=True)
np.save("data_blocks/train_x.npy", train_x)
np.save("data_blocks/train_y.npy", train_y)
np.save("data_blocks/valid_x.npy", valid_x)
np.save("data_blocks/valid_y.npy", valid_y)

print("Guardado en ./data_blocks (train_x/y.npy, valid_x/y.npy)")

