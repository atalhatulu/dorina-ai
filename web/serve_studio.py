import os
import sys
import json
import time
import asyncio
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Root path configuration
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from model_moe_v3 import DorinaMoEV3, MoEConfig

app = FastAPI(title="Dorina AI Studio", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOK_PATH = os.path.join(ROOT_DIR, "tokenizers", "dorina_tokenizer_v3_32k.json")

# Model Registry
AVAILABLE_MODELS = {
    "v3_sft_base": {
        "name": "Dorina V3 SFT Base (Temiz & Doğal Model)",
        "score": 60.5,
        "path": os.path.join(ROOT_DIR, "kaggle_upload", "dorina-v3-checkpoint", "dorina_v3_sft_base.pt"),
        "desc": "180.6M MoE. Zehirlenmemiş, temiz SFT modeli (%87.5 Algoritma başarısı).",
        "params": "180.6M (Aktif: 40M)"
    },
    "v3_titan_4500": {
        "name": "Dorina V3 Titan Step 4500 (Pre-train Zirve)",
        "score": 51.8,
        "path": os.path.join(ROOT_DIR, "kaggle_upload", "dorina-v3-checkpoint", "dorina_v3_step_4500.pt"),
        "desc": "1.19 Milyar token pre-train edilmiş saf tamamlayıcı model.",
        "params": "180.6M (Aktif: 40M)"
    }
}

CURRENT_MODEL_KEY = "v3_sft_base"
CURRENT_MODEL = None
TOKENIZER = None

def load_engine(model_key: str):
    global CURRENT_MODEL, TOKENIZER, CURRENT_MODEL_KEY
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Geçersiz model: {model_key}")
    
    info = AVAILABLE_MODELS[model_key]
    model_path = info["path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
    print(f"[Dorina Studio] Yükleniyor: {info['name']} -> {DEVICE}...")
    
    if TOKENIZER is None:
        TOKENIZER = Tokenizer.from_file(TOK_PATH)
        
    cfg = MoEConfig(
        vocab_size=32768,
        n_layer=12,
        d_model=512,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=1024,
        n_experts=32,
        top_k=2,
        expert_hidden=256,
        shared_hidden=256
    )
    
    model = DorinaMoEV3(cfg)
    sd = torch.load(model_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd)
    model.to(DEVICE)
    model.eval()
    
    CURRENT_MODEL = model
    CURRENT_MODEL_KEY = model_key
    print(f"[Dorina Studio] {info['name']} başarıyla hazırlandı!")

# Initial Load
try:
    load_engine(CURRENT_MODEL_KEY)
except Exception as e:
    print(f"[UYARI] Başlangıç modeli yüklenemedi: {e}")

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.5
    top_k: int = 25
    max_tokens: int = 150
    system_prompt: Optional[str] = "Sen Talha tarafından MoE mimarisiyle sıfırdan eğitilmiş Dorina AI adında zeki bir asistansın."

@app.get("/api/models")
def get_models():
    return {
        "current": CURRENT_MODEL_KEY,
        "models": AVAILABLE_MODELS
    }

@app.post("/api/models/switch/{model_key}")
def switch_model(model_key: str):
    try:
        load_engine(model_key)
        return {"status": "success", "current": CURRENT_MODEL_KEY, "info": AVAILABLE_MODELS[model_key]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/benchmarks")
def get_benchmarks():
    eval_dir = os.path.join(ROOT_DIR, "eval_results")
    results = {}
    if os.path.exists(eval_dir):
        for f in os.listdir(eval_dir):
            if f.endswith(".json"):
                p = os.path.join(eval_dir, f)
                try:
                    with open(p, "r", encoding="utf-8") as jf:
                        results[f] = json.load(jf)
                except Exception:
                    pass
    return results

@app.get("/api/metrics")
def get_system_metrics():
    vram_alloc = torch.cuda.memory_allocated(DEVICE) / (1024**2) if torch.cuda.is_available() else 0
    vram_res = torch.cuda.memory_reserved(DEVICE) / (1024**2) if torch.cuda.is_available() else 0
    moe_health = CURRENT_MODEL.moe_metrics() if CURRENT_MODEL else {}
    return {
        "device": DEVICE,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_allocated_mb": round(vram_alloc, 1),
        "vram_reserved_mb": round(vram_res, 1),
        "current_model": CURRENT_MODEL_KEY,
        "moe_health": moe_health
    }

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    if CURRENT_MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model yüklü değil")
        
    full_prompt = (
        f"<|im_start|>system\n{req.system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{req.message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    input_ids = TOKENIZER.encode(full_prompt).ids
    idx = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    gen_tokens = []
    
    t0 = time.time()
    with torch.no_grad():
        for _ in range(req.max_tokens):
            idx_cond = idx if idx.size(1) <= 512 else idx[:, -512:]
            logits, _ = CURRENT_MODEL(idx_cond)
            logits = logits[0, -1, :].clone()
            
            # Repetition Penalty (Tekrar ve döngü engelleme)
            rep_pen = 1.2
            for seen_tok in set(input_ids + gen_tokens):
                if logits[seen_tok] > 0:
                    logits[seen_tok] /= rep_pen
                else:
                    logits[seen_tok] *= rep_pen
                    
            logits = logits / max(req.temperature, 1e-4)
            if req.top_k is not None:
                v, _ = torch.topk(logits, min(req.top_k, logits.size(-1)))
                logits[logits < v[[-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1).item()
            if next_t == 3: # <|im_end|>
                break
            gen_tokens.append(next_t)
            idx = torch.cat([idx, torch.tensor([[next_t]], device=DEVICE)], dim=1)
            
    dt = time.time() - t0
    tok_speed = len(gen_tokens) / max(dt, 1e-4)
    response_text = TOKENIZER.decode(gen_tokens).strip()
    
    # Canlı MoE aktivasyonu
    moe_health = CURRENT_MODEL.moe_metrics()
    
    return {
        "reply": response_text,
        "tokens_generated": len(gen_tokens),
        "duration_sec": round(dt, 3),
        "speed_tok_s": round(tok_speed, 1),
        "moe_health": moe_health
    }

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            req_data = json.loads(data)
            user_msg = req_data.get("message", "")
            temp = float(req_data.get("temperature", 0.4))
            top_k = int(req_data.get("top_k", 25))
            max_tokens = int(req_data.get("max_tokens", 200))
            
            full_prompt = (
                f"<|im_start|>system\nSen Talha tarafından geliştirilen 180.6M MoE mimarili Dorina V3 yapay zeka asistanısın.<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            input_ids = TOKENIZER.encode(full_prompt).ids
            idx = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
            
            t0 = time.time()
            total_tokens = 0
            with torch.no_grad():
                for _ in range(max_tokens):
                    idx_cond = idx if idx.size(1) <= 512 else idx[:, -512:]
                    logits, _ = CURRENT_MODEL(idx_cond)
                    logits = logits[:, -1, :] / max(temp, 1e-4)
                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('Inf')
                    probs = F.softmax(logits, dim=-1)
                    next_t = torch.multinomial(probs, num_samples=1).item()
                    if next_t == 3: # <|im_end|>
                        break
                    
                    total_tokens += 1
                    token_str = TOKENIZER.decode([next_t])
                    idx = torch.cat([idx, torch.tensor([[next_t]], device=DEVICE)], dim=1)
                    
                    # Token streaming ve uzman istatistiği
                    await websocket.send_json({
                        "type": "token",
                        "token": token_str
                    })
                    await asyncio.sleep(0.005) # Akıcı UI deneyimi
                    
            dt = time.time() - t0
            await websocket.send_json({
                "type": "done",
                "total_tokens": total_tokens,
                "duration_sec": round(dt, 2),
                "speed_tok_s": round(total_tokens / max(dt, 1e-4), 1)
            })
    except WebSocketDisconnect:
        pass

@app.get("/")
def serve_index():
    with open(os.path.join(ROOT_DIR, "web", "studio.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    uvicorn.run("serve_studio:app", host="127.0.0.1", port=8000, reload=False)
