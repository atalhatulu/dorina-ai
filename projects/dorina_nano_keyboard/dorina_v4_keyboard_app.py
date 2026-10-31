import os
import sys
import re
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from tokenizers import Tokenizer
from kaggle_v4_shard5_kernel.main import DorinaDenseV4, DenseConfig

app = FastAPI(title="Dorina-V4 Akıllı Klavye ve Sonraki Kelime Tahmin Motoru")

device = "cuda" if torch.cuda.is_available() else "cpu"
tok_path = "tokenizers/dorina_tokenizer_v4_32k.json"
tokenizer = Tokenizer.from_file(tok_path)

cfg = DenseConfig(vocab_size=32768, max_seq_len=512)
model = DorinaDenseV4(cfg)
weights_path = "models/dorina_v4/dorina_v4_shard6_1to20_crystal_final.pt"
sd = torch.load(weights_path, map_location="cpu", weights_only=False)
model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True)
model.to(device)
model.eval()

print(f"[+] Dorina-V4 144.7M Kristal Model ({device.upper()}) Akilli Klavye Motoru Aktif!")

class PredictRequest(BaseModel):
    text: str
    top_k: int = 5

def generate_continuation_word(base_ids, first_token_id, max_subwords=4):
    curr = list(base_ids) + [first_token_id]
    word_tokens = [first_token_id]
    
    for _ in range(max_subwords):
        inp = torch.tensor([curr[-256:]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(inp)
            logits = out[0] if isinstance(out, tuple) else out
            next_t = torch.argmax(logits[0, -1, :]).item()
        
        piece = tokenizer.decode([next_t])
        if piece.startswith(" ") or piece in [".", ",", "!", "?", "\n"] or next_t in [0, 2]:
            break
        word_tokens.append(next_t)
        curr.append(next_t)
        
    full_word = tokenizer.decode(word_tokens).strip()
    return full_word

@app.post("/predict")
def predict_next_tokens(req: PredictRequest):
    raw_text = req.text.strip()
    if not raw_text:
        return {"predictions": []}

    enc = tokenizer.encode(raw_text)
    if not enc.ids:
        return {"predictions": []}

    # Extract recently used words to penalize repetition (Repetition Penalty)
    words_in_text = [w.lower().strip(".,!?:;\"'()[]{}") for w in raw_text.split()]
    recent_words = set(words_in_text[-6:]) # son 6 kelimeyi yasaklı havuzuna al

    input_ids = torch.tensor([enc.ids[-256:]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(input_ids)
        logits = out[0] if isinstance(out, tuple) else out
        last_logits = logits[0, -1, :].clone()

        # Apply repetition penalty directly to recently seen token IDs
        for tid in set(enc.ids[-16:]):
            if tid < 32768:
                last_logits[tid] -= 3.5 # Ağır tekrar cezası

        probs = F.softmax(last_logits, dim=-1)

    topk_probs, topk_indices = torch.topk(probs, 60)

    suggestions = []
    seen = set()

    for prob, idx in zip(topk_probs, topk_indices):
        tid = idx.item()
        if tid in [0, 1, 2]:
            continue

        raw_piece = tokenizer.decode([tid])
        if not raw_piece.strip():
            continue

        full_word = generate_continuation_word(enc.ids, tid)
        clean = full_word.strip()
        clean_lower = clean.lower()

        # REPETITION FILTER: Daha önce geçen kelimeleri veya anlamsız kısa harfleri ele!
        if (clean_lower not in recent_words and 
            clean_lower not in seen and 
            len(clean) > 1 and 
            not re.match(r'^[0-9]+$', clean)):
            
            seen.add(clean_lower)
            suggestions.append({
                "word": clean,
                "confidence": round(prob.item() * 100, 1)
            })

        if len(suggestions) >= req.top_k:
            break

    return {"predictions": suggestions}

@app.get("/", response_class=HTMLResponse)
def get_keyboard_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dorina-V4 144M Akıllı Klavye</title>
        <style>
            body {
                background: #090d16;
                color: #f1f5f9;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                margin: 0;
                padding: 16px;
            }
            .container {
                width: 100%;
                max-width: 620px;
                background: #131b2e;
                border-radius: 20px;
                padding: 26px;
                box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.7);
                border: 1px solid #1e293b;
            }
            h2 {
                margin-top: 0;
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 24px;
                color: #38bdf8;
            }
            .info {
                font-size: 13px;
                color: #94a3b8;
                margin-bottom: 20px;
                line-height: 1.5;
            }
            /* SwiftKey / Gboard Style Suggestion Bar */
            .suggestions-bar {
                display: flex;
                gap: 8px;
                margin-bottom: 16px;
                min-height: 52px;
                background: #090d16;
                padding: 6px 10px;
                border-radius: 12px;
                border: 1px solid #334155;
                align-items: center;
                justify-content: flex-start;
                overflow-x: auto;
            }
            .suggestion-chip {
                background: #1e293b;
                color: #f8fafc;
                padding: 9px 16px;
                border-radius: 9px;
                cursor: pointer;
                font-size: 15px;
                font-weight: 500;
                display: flex;
                align-items: center;
                gap: 8px;
                transition: all 0.15s ease;
                white-space: nowrap;
                border: 1px solid #334155;
            }
            .suggestion-chip:hover {
                background: #0284c7;
                border-color: #38bdf8;
                transform: translateY(-2px);
            }
            .suggestion-chip span.conf {
                font-size: 10px;
                color: #7dd3fc;
                background: rgba(0,0,0,0.35);
                padding: 2px 6px;
                border-radius: 5px;
            }
            textarea {
                width: 100%;
                height: 150px;
                background: #090d16;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 16px;
                color: #f8fafc;
                font-size: 18px;
                resize: none;
                box-sizing: border-box;
                outline: none;
                font-family: inherit;
                line-height: 1.5;
            }
            textarea:focus {
                border-color: #38bdf8;
                box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25);
            }
            .stats {
                display: flex;
                justify-content: space-between;
                margin-top: 14px;
                font-size: 13px;
                color: #64748b;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>⌨️ Dorina-V4 (144M) Akıllı Klavye</h2>
            <div class="info">Tekrar engelleyici filtre devrede! Yazmaya başla; model sıradaki kelimeleri üst çubuğa önersin. Tıklayarak veya <b>TAB</b> tuşuna basarak metne ekle!</div>

            <!-- Suggestion Bar -->
            <div class="suggestions-bar" id="suggestions">
                <div style="color: #64748b; font-size: 13px; padding-left: 8px;">Kelimeler bekleniyor...</div>
            </div>

            <textarea id="editor" placeholder="Buraya yaz (Örn: Türkiye'nin başkenti, Yapay zeka teknolojileri, Bugün hava)..." autofocus></textarea>

            <div class="stats">
                <span>Model: Dorina-V4 (144.7M Dense Kristal)</span>
                <span id="latency">Gecikme: 0 ms</span>
            </div>
        </div>

        <script>
            const editor = document.getElementById('editor');
            const suggestionsBar = document.getElementById('suggestions');
            const latencySpan = document.getElementById('latency');
            let debounceTimer = null;

            async function fetchPredictions() {
                const text = editor.value;
                if (!text.trim()) {
                    suggestionsBar.innerHTML = '<div style="color: #64748b; font-size: 13px; padding-left: 8px;">Kelimeler bekleniyor...</div>';
                    return;
                }

                const t0 = performance.now();
                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: text, top_k: 5})
                    });
                    const data = await res.json();
                    const t1 = performance.now();
                    latencySpan.innerText = 'Gecikme: ' + Math.round(t1 - t0) + ' ms';

                    if (data.predictions && data.predictions.length > 0) {
                        suggestionsBar.innerHTML = '';
                        data.predictions.forEach(p => {
                            const btn = document.createElement('div');
                            btn.className = 'suggestion-chip';
                            btn.innerHTML = `${p.word} <span class="conf">%${p.confidence}</span>`;
                            btn.onclick = () => {
                                appendWord(p.word);
                            };
                            suggestionsBar.appendChild(btn);
                        });
                    }
                } catch (e) {
                    console.error(e);
                }
            }

            function appendWord(word) {
                let current = editor.value;
                if (!current.endsWith(' ')) {
                    current += ' ';
                }
                current += word + ' ';
                editor.value = current;
                editor.focus();
                fetchPredictions();
            }

            editor.addEventListener('input', () => {
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(fetchPredictions, 60);
            });

            // TAB key takes first suggestion
            editor.addEventListener('keydown', (e) => {
                if (e.key === 'Tab') {
                    const firstChip = suggestionsBar.querySelector('.suggestion-chip');
                    if (firstChip) {
                        e.preventDefault();
                        firstChip.click();
                    }
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    print("[*] Dorina-V4 Klavye Uygulamasi http://127.0.0.1:8501 adresinde baslatiliyor...")
    uvicorn.run(app, host="0.0.0.0", port=8501)
