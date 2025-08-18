import torch
from transformers import AutoTokenizer
from train_minigpt import GPT, GPTConfig

TOKENIZER_DIR = "tokenizer_gpt2_es"
CHECKPOINT_PATH = "gpt_minimo.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
cfg = GPTConfig(**ckpt["config"])
model = GPT(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

@torch.no_grad()
def generate(model, start_ids, max_new_tokens=80, temperature=0.9, top_k=50):
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, :]
    for _ in range(max_new_tokens):
        logits, _ = model(x[:, -model.cfg.n_positions:])
        logits = logits[:, -1, :] / max(1e-6, temperature)
        if top_k:
            v, _ = torch.topk(logits, k=top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
    return x[0].tolist()

print("MiniGPT listo. Escribe un prompt y ENTER (vacío para salir).")
while True:
    prompt = input(">>> ")
    if not prompt.strip():
        break
    seed = tok.encode(prompt)
    out_ids = generate(model, seed, max_new_tokens=80, temperature=0.9, top_k=50)
    print(tok.decode(out_ids))
