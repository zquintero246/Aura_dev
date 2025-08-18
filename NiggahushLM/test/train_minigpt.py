import os, math, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer

# --- Rutas de tus datos y tokenizer ---
DATA_DIR = "data_blocks"              # Ruta de los datos apilados despues del preprocesamiento
TOKENIZER_DIR = "tokenizer_gpt2_es"   # El tokenizer cacheado para futuro escalamiento (i guess)

# --- Semilla para resultados reproducibles ---
torch.manual_seed(1337)
np.random.seed(1337)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# --- Se carga el tokenizer que guardamos para hacer cositas y asi ---
tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.model_max_length = 10**9
vocab_size = tok.vocab_size
print("Vocab size:", vocab_size)

# --- Se leen los .npy y se define un dataset simplecito que entrega x, y ya listos ---

class NpyBlocks(Dataset):
    def __init__(self, x_path, y_path):
        # mmap evita cargar TODO en RAM de una, asi no se me estalla la ram
        self.x = np.load(x_path, mmap_mode='r')   # shape: [N, T] int32
        self.y = np.load(y_path, mmap_mode='r')   # shape: [N, T] int32
        assert self.x.shape == self.y.shape, "x/y shapes no coinciden"

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        # Embedding en PyTorch espera int64 (long)
        x = torch.from_numpy(self.x[i].astype(np.int64))
        y = torch.from_numpy(self.y[i].astype(np.int64))
        return x, y

train_ds = NpyBlocks(os.path.join(DATA_DIR, "train_x.npy"),
                     os.path.join(DATA_DIR, "train_y.npy"))
valid_ds = NpyBlocks(os.path.join(DATA_DIR, "valid_x.npy"),
                     os.path.join(DATA_DIR, "valid_y.npy"))

# En Windows, num_workers=0 para evitar líos
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  drop_last=True,  pin_memory=True, num_workers=0)
valid_loader = DataLoader(valid_ds, batch_size=4, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)

print("Batches train:", len(train_loader), " | Batches valid:", len(valid_loader))

# ==================================== PARTE REALMENTE CHIMBA ===============================================


# Se define la arquitectura de un gpt chiquitico

class GPTConfig:
    def __init__(self, vocab_size, n_layer=6, n_head=8, d_model=512, n_positions=256, dropout=0.1):
        self.vocab_size=vocab_size; self.n_layer=n_layer; self.n_head=n_head
        self.d_model=d_model; self.n_positions=n_positions; self.dropout=dropout

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.nh = cfg.n_head
        self.hs = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3*cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        def split_heads(t):  # [B,T,C] -> [B,nh,T,hs]
            return t.view(B, T, self.nh, self.hs).transpose(1, 2)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        # atencion con mascara causal integrada
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop.p, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # [B,T,C]
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, 4*cfg.d_model)
        self.fc2 = nn.Linear(4*cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp  = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.n_positions, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)  # [1,T]
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B,T,vocab]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

def count_params(model):
    return sum(p.numel() for p in model.parameters())/1e6

# ======================= se instancia el modelo y el optimizador ================================

# Usa el mismo block_size que tus datos (256)
cfg = GPTConfig(vocab_size=vocab_size, n_layer=6, n_head=8, d_model=512,
                n_positions=256, dropout=0.1)
model = GPT(cfg).to(device)
print(f"Parámetros totales: {count_params(model):.2f}M")

opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))


# ======================= Flujo de entrenamiento ================================

def run_epoch(dl, train=True):
    model.train(train)
    total, n = 0.0, 0
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            _, loss = model(x, y)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        total += loss.item(); n += 1
    return total / max(1, n)

best_val = float("inf")
EPOCHS = 3  # empieza cortito

for epoch in range(EPOCHS):
    tr = run_epoch(train_loader, train=True)
    va = run_epoch(valid_loader, train=False)
    ppl = math.exp(va)
    print(f"epoch {epoch} | train {tr:.4f} | valid {va:.4f} | ppl {ppl:.1f}")
    if va < best_val:
        best_val = va
        torch.save({"model": model.state_dict(), "config": vars(cfg)}, "gpt_minimo.pt")
        print(">> guardado: gpt_minimo.pt")

# ======================= Testing ================================

@torch.no_grad()
def generate(model, start_ids, max_new_tokens=60, temperature=1.0, top_k=50):
    model.eval()
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, :]
    for _ in range(max_new_tokens):
        logits, _ = model(x[:, -model.cfg.n_positions:])  # recorta al contexto
        logits = logits[:, -1, :] / max(1e-6, temperature)
        if top_k:
            v, _ = torch.topk(logits, k=top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
    return x[0].tolist()

seed = tok.encode("En el parque confuso")
out_ids = generate(model, seed, max_new_tokens=80, temperature=0.9, top_k=50)
print(tok.decode(out_ids))




