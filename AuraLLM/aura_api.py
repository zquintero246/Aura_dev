from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
import time
import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/aura_Q6.gguf")
N_THREADS = int(os.getenv("N_THREADS", 8))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", 50))
CTX_SIZE = int(os.getenv("CTX_SIZE", 32768))

print(f"Cargando AuraLM desde {MODEL_PATH} ...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CTX_SIZE,
    n_threads=N_THREADS,
    n_gpu_layers=N_GPU_LAYERS,
    verbose=False
)
print("Modelo AuraLM listo.")

app = FastAPI(title="Aura Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost", "https://aura.dev"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "model": "AuraLM", "ctx": CTX_SIZE}

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    max_tokens = int(data.get("max_tokens", 256))
    temperature = float(data.get("temperature", 0.7))

    start = time.time()
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|end|>", "</s>", "User:", "Assistant:"],
    )
    latency = int((time.time() - start) * 1000)

    text = output["choices"][0]["text"].strip()
    return {"output": text, "latency_ms": latency}
