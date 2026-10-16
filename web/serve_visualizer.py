import os
import sys
import glob
import torch
import torch.nn.functional as F
import tiktoken
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Proje kök dizinini Python yoluna ekle
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.model import DorinaGPT

app = FastAPI(title="Dorina Neural Visualizer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

# Model Konfigürasyonu (70M)
CFG = {"n_layer": 10, "n_embd": 512, "n_head": 8, "context_length": 256}

model = DorinaGPT(
    vocab_size=enc.n_vocab,
    context_length=CFG["context_length"],
    n_embd=CFG["n_embd"],
    n_head=CFG["n_head"],
    n_layer=CFG["n_layer"],
    dropout=0.0
).to(device)

# En güncel checkpoint dosyasını kök dizinden itibaren ara
runs_path_pattern = os.path.join(ROOT_DIR, "runs", "dorina_v0.13_70m", "run*", "*.pt")
found_files = sorted(glob.glob(runs_path_pattern))

# Öncelik best.pt, yoksa model.pt
load_path = None
best_files = [f for f in found_files if os.path.basename(f) == "best.pt"]
model_files = [f for f in found_files if os.path.basename(f) == "model.pt"]

if best_files:
    load_path = best_files[-1]
elif model_files:
    load_path = model_files[-1]

if load_path and os.path.exists(load_path):
    print(f"\n📦 Görselleştirici için model yükleniyor: {load_path}")
    ckpt = torch.load(load_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    print("✅ 70M Model ağırlıkları başarıyla yüklendi ve aktif!\n")
else:
    print(f"\n❌ DİKKAT: Eğitilmiş ağırlık bulunamadı! Taranan yer: {runs_path_pattern}")
    print("Model rastgele başlatılmış ağırlıklarla çalışıyor.\n")

model.eval()

class InspectRequest(BaseModel):
    text: str
    top_k: int = 6

@app.post("/inspect")
@torch.no_grad()
def inspect_step(req: InspectRequest):
    if not req.text.strip():
        return {"tokens": [], "predictions": []}

    tokens = enc.encode(req.text)
    tokens = tokens[-CFG["context_length"]:]
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        logits, _ = model(idx)
    
    last_logits = logits[0, -1, :]
    probs = F.softmax(last_logits, dim=-1)
    
    top_probs, top_indices = torch.topk(probs, req.top_k)
    
    predictions = []
    for p, i in zip(top_probs.tolist(), top_indices.tolist()):
        word = enc.decode([i])
        predictions.append({
            "token_id": i,
            "text": word,
            "probability": round(p * 100, 2)
        })

    return {
        "tokens": [enc.decode([t]) for t in tokens],
        "predictions": predictions,
        "token_count": len(tokens)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)