import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from scripts.dorina_nano_tr_model import DorinaNanoTR, NanoTRConfig
from tokenizers import Tokenizer
import re

app = FastAPI(title="Dorina Nano Production-Grade Keyboard Engine")

MODEL_PATH = "models/dorina_nano_tr/dorina_nano_tr_10m.pt"
TOKENIZER_PATH = "tokenizers/dorina_nano_tr_12k.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

cfg = NanoTRConfig(
    vocab_size=12288,
    max_seq_len=256,
    d_model=256,
    n_layer=10,
    n_heads=8,
    n_kv_heads=4,
    head_dim=32,
    intermediate_size=688,
    tied_embeddings=True
)

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model = DorinaNanoTR(cfg).to(device)
model.load_state_dict(state_dict)
model.eval()

# #5 & #6: Kapsamlı Soru Eki ve Bağlaç Haritası (Büyük/Küçük Harf Korumalı)
MI_SUFFIXES = [
    "misiniz", "mısınız", "musunuz", "müsünüz",
    "miyiz", "mıyız", "muyuz", "müyüz",
    "misin", "mısın", "musun", "müsün",
    "miydi", "mıydı", "muydu", "müydü",
    "miymiş", "mıymış", "muymuş", "müymüş",
    "mı", "mi", "mu", "mü"
]
# Uzundan kısaya sırala ki 'miyiz' önce yakalansın, 'mi' sonra
MI_SUFFIXES_SORTED = sorted(MI_SUFFIXES, key=len, reverse=True)
MI_SUFFIX_REGEX = re.compile(rf'([a-zA-ZçğıöşüÇĞİÖŞÜ]+)({"|".join(MI_SUFFIXES_SORTED)})$', re.IGNORECASE)

DEDA_SPLIT_MAP = {
    "bende": "ben de",
    "sende": "sen de",
    "bizde": "biz de",
    "sizde": "siz de",
    "onlarda": "onlar da"
}

def clean_punctuation(text):
    return text.rstrip('.!?,;: ')

def preserve_case(source, target):
    # #6: Büyük/Küçük harf adaptasyonu (örn: Bende -> Ben de, Yarin -> Yarın)
    if not source or not target:
        return target
    if source.isupper():
        return target.upper()
    if source[0].isupper():
        return target[0].upper() + target[1:]
    return target

def fix_word_strictly(word):
    raw_word = clean_punctuation(word.strip())
    if not raw_word:
        return word
    
    lower_w = raw_word.lower()
    
    # 1. Bağlaç Ayrımı
    if lower_w in DEDA_SPLIT_MAP:
        return preserve_case(raw_word, DEDA_SPLIT_MAP[lower_w])
        
    # 2. Soru Eki Ayrımı
    exceptions = {'cami', 'gemici', 'demirci', 'vergi', 'sevgi', 'bilgi', 'ilmi', 'resmi', 'mermi'}
    if lower_w not in exceptions:
        m = MI_SUFFIX_REGEX.match(lower_w)
        if m:
            stem, suffix = m.groups()
            if len(stem) >= 3:
                return preserve_case(raw_word, f"{stem} {suffix}")

    # 3. Model ile Düzeltme
    fix_prompt = f"<fix> {raw_word} =>"
    input_ids = tokenizer.encode(fix_prompt).ids
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    gen = []
    
    with torch.no_grad():
        for _ in range(16):
            res = model(x)
            logits = res[0] if isinstance(res, tuple) else res
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tok_id = next_id.item()
            if tok_id == tokenizer.token_to_id("<eos>") or tok_id == 3:
                break
            # #1: Asla tek token'ı decode edip '\n' kontrolü yapmıyoruz, UTF-8 byte kırılır!
            gen.append(tok_id)
            x = torch.cat((x, next_id), dim=1)
            
    # #1: Tüm token listesini topluca decode et (Byte-level BPE Türkçe harfleri sağlam kalır)
    fixed_full = tokenizer.decode(gen).strip()
    if '\n' in fixed_full:
        fixed_full = fixed_full.split('\n')[0].strip()
    fixed_full = clean_punctuation(fixed_full)
    
    # #2: Eğer model soru eki veya bağlaç için 2 kelimeye ayırdıysa ('ben de', 'yapabilir miyiz') KESME, KORU!
    tokens = fixed_full.split()
    if len(tokens) == 2 and (tokens[1].lower() in MI_SUFFIXES or tokens[1].lower() in ["de", "da"]):
        return preserve_case(raw_word, fixed_full)
    elif len(tokens) >= 1:
        return preserve_case(raw_word, tokens[0])
        
    return raw_word

def get_word_completions(prefix_text, max_candidates=3):
    if not prefix_text:
        return []
        
    # #3: Trailing space temizliği. Eğer kullanıcı kelime yazıyorsa o kelimeye odaklan!
    has_trailing_space = prefix_text.endswith(" ")
    words = prefix_text.strip().split()
    current_typing = "" if has_trailing_space else (words[-1] if words else "")
    
    candidates = []
    
    # A) İmla/Düzeltme Adayı (Yazılan kelimenin kendisi)
    if len(current_typing) >= 2:
        corrected_self = fix_word_strictly(current_typing)
        if corrected_self and corrected_self.lower() != current_typing.lower():
            candidates.append({"word": corrected_self, "type": "Düzeltme"})
            
    # B) Model Tamamlama Tahminleri
    full_prompt = f"<complete> {prefix_text}"
    input_ids = tokenizer.encode(full_prompt).ids
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        res = model(x)
        logits = res[0] if isinstance(res, tuple) else res
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top_probs, top_indices = torch.topk(probs, 25)
        
    for idx in top_indices:
        tok_id = idx.item()
        # #1: Güvenli decode
        tok_str = tokenizer.decode([tok_id])
        if not tok_str.strip() or tok_str.startswith("<"):
            continue
            
        if not tok_str.startswith(" ") and current_typing:
            full_completed = (current_typing + tok_str).strip(' .,!?:;\n')
        else:
            full_completed = tok_str.strip(' .,!?:;\n')
            
        full_completed = clean_punctuation(full_completed)
        
        if full_completed and not any(c["word"].lower() == full_completed.lower() for c in candidates):
            candidates.append({"word": full_completed, "type": "Öneri"})
            
        if len(candidates) >= max_candidates:
            break
            
    return candidates[:max_candidates]

class PredictRequest(BaseModel):
    text: str

class WordRequest(BaseModel):
    word: str

@app.post("/api/candidates")
def api_candidates(req: PredictRequest):
    candidates = get_word_completions(req.text, max_candidates=3)
    return {"candidates": candidates}

@app.post("/api/fix_word")
def api_fix_word(req: WordRequest):
    fixed = fix_word_strictly(req.word)
    return {"fixed": fixed}

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>Dorina Mobile Keyboard (Production Grade)</title>
    <style>
        * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: #070b14; color: #f8fafc; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; margin: 0; padding: 20px; }
        .card { background: #0f172a; border-radius: 24px; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.8); width: 100%; max-width: 580px; padding: 25px; border: 1px solid #1e293b; }
        
        .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; }
        .header h1 { margin: 0; font-size: 20px; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
        .badge { background: #0284c7; color: #fff; font-size: 11px; padding: 4px 10px; border-radius: 20px; font-weight: bold; }
        
        .keyboard-bar {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 8px;
            background: #1e293b;
            padding: 8px;
            border-radius: 14px;
            margin-bottom: 12px;
            border: 1px solid #334155;
            min-height: 52px;
        }
        .candidate-slot {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 10px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 6px 10px;
            cursor: pointer;
            transition: all 0.15s ease;
            user-select: none;
            text-align: center;
        }
        .candidate-slot:hover {
            background: #38bdf8;
            color: #0f172a;
            border-color: #38bdf8;
            transform: translateY(-2px);
        }
        .candidate-slot .word {
            font-size: 16px;
            font-weight: 600;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 100%;
        }
        .candidate-slot .tag {
            font-size: 10px;
            opacity: 0.7;
            text-transform: uppercase;
            margin-top: 2px;
        }
        .candidate-slot.center {
            border-color: #38bdf8;
            background: #132238;
        }
        .empty-slot {
            opacity: 0.2;
            cursor: default;
        }
        .empty-slot:hover {
            background: #0f172a;
            color: inherit;
            transform: none;
            border-color: #334155;
        }

        .input-box {
            width: 100%;
            height: 120px;
            background: #080d1a;
            border: 2px solid #334155;
            border-radius: 16px;
            color: #fff;
            font-size: 19px;
            padding: 15px;
            resize: none;
            outline: none;
            transition: all 0.2s;
            line-height: 1.6;
        }
        .input-box:focus {
            border-color: #38bdf8;
            box-shadow: 0 0 15px rgba(56,189,248,0.25);
        }

        .info-pill {
            background: #090e1a;
            padding: 10px 14px;
            border-radius: 12px;
            font-size: 12px;
            color: #94a3b8;
            border: 1px solid #1e293b;
            margin-top: 15px;
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .footer-info {
            margin-top: 15px;
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #64748b;
        }
        .fix-notice {
            font-size: 13px;
            color: #38bdf8;
            min-height: 20px;
            margin-top: 6px;
            font-weight: 500;
        }
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <h1><span>📱 Dorina Mobil Klavye</span></h1>
            <span class="badge">1.0B TOKEN MOTOR</span>
        </div>

        <div id="keyboardBar" class="keyboard-bar">
            <div id="slot0" class="candidate-slot empty-slot" onclick="selectSlot(0)">
                <span class="word">-</span>
            </div>
            <div id="slot1" class="candidate-slot center empty-slot" onclick="selectSlot(1)">
                <span class="word">-</span>
            </div>
            <div id="slot2" class="candidate-slot empty-slot" onclick="selectSlot(2)">
                <span class="word">-</span>
            </div>
        </div>

        <textarea id="editor" class="input-box" placeholder="Yazmaya başlayın (Örn: 'yar', 'yard', 'gel', 'bunu yapabilirmiyiz')..."></textarea>

        <div id="fixNotice" class="fix-notice"></div>

        <div class="info-pill">
            <span>🛡️</span>
            <span><b>Düzeltmeler Entegre Edildi:</b> Byte-level UTF-8 güvenliği, imleç ortasındayken doğru kelime yakalama, senkronize boşluk ve çok kelimeli soru eki koruması aktif.</span>
        </div>

        <div class="footer-info">
            <div>🟢 Durum: Üretim Seviyesi Klavye Motoru</div>
            <div>Gecikme: <span id="latency">0</span> ms</div>
        </div>
    </div>

    <script>
        const editor = document.getElementById('editor');
        const latencyBox = document.getElementById('latency');
        const fixNotice = document.getElementById('fixNotice');

        let currentCandidates = [];
        let debounceTimer;

        editor.addEventListener('input', () => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(updateCandidates, 25);
        });

        async function updateCandidates() {
            // #7: İmlecin bulunduğu konuma göre o anki metni al
            const cursorPos = editor.selectionStart;
            const textToCursor = editor.value.slice(0, cursorPos);
            
            if (!textToCursor.trim()) {
                clearSlots();
                return;
            }

            const t0 = performance.now();
            try {
                const res = await fetch('/api/candidates', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: textToCursor })
                });
                const data = await res.json();
                latencyBox.innerText = Math.round(performance.now() - t0);

                currentCandidates = data.candidates || [];
                renderSlots(currentCandidates);
            } catch (err) {}
        }

        function renderSlots(candidates) {
            for (let i = 0; i < 3; i++) {
                const slot = document.getElementById(`slot${i}`);
                if (candidates[i]) {
                    slot.className = `candidate-slot ${i === 1 ? 'center' : ''}`;
                    slot.innerHTML = `<span class="word">${candidates[i].word}</span><span class="tag">${candidates[i].type}</span>`;
                } else {
                    slot.className = `candidate-slot ${i === 1 ? 'center' : ''} empty-slot`;
                    slot.innerHTML = `<span class="word">-</span>`;
                }
            }
        }

        function clearSlots() {
            currentCandidates = [];
            for (let i = 0; i < 3; i++) {
                const slot = document.getElementById(`slot${i}`);
                slot.className = `candidate-slot ${i === 1 ? 'center' : ''} empty-slot`;
                slot.innerHTML = `<span class="word">-</span>`;
            }
        }

        // #7: İmleç konumuna göre kelime değiştirme
        function selectSlot(index) {
            if (!currentCandidates[index]) return;
            const chosen = currentCandidates[index].word;

            const fullVal = editor.value;
            const cursorPos = editor.selectionStart;
            const before = fullVal.slice(0, cursorPos);
            const after = fullVal.slice(cursorPos);

            const match = before.match(/\\S+$/);
            if (match) {
                const wordStart = match.index;
                editor.value = before.slice(0, wordStart) + chosen + ' ' + after;
                const newPos = wordStart + chosen.length + 1;
                editor.setSelectionRange(newPos, newPos);
            } else {
                editor.value = before + chosen + ' ' + after;
                const newPos = before.length + chosen.length + 1;
                editor.setSelectionRange(newPos, newPos);
            }
            editor.focus();
            updateCandidates();
        }

        // #4: Senkron Space Düzeltmesi (Event Loop çakışması ve çift boşluk önlendi)
        editor.addEventListener('keydown', (e) => {
            if (e.key === ' ') {
                const fullVal = editor.value;
                const cursorPos = editor.selectionStart;
                const before = fullVal.slice(0, cursorPos);
                const after = fullVal.slice(cursorPos);

                // #7: İmlecin hemen solundaki kelimeyi güvenli regex ile al
                const match = before.match(/([a-zA-ZçğıöşüÇĞİÖŞÜ]+)$/);
                if (match) {
                    const targetWord = match[1];
                    if (targetWord.length >= 2) {
                        e.preventDefault(); // #4: Hemen durdur ki çift boşluk veya asenkron yarış olmasın
                        
                        fetch('/api/fix_word', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ word: targetWord })
                        })
                        .then(r => r.json())
                        .then(data => {
                            const fixed = data.fixed || targetWord;
                            const wordStart = match.index;
                            
                            // Yeni metin montajı
                            editor.value = before.slice(0, wordStart) + fixed + ' ' + after;
                            const newPos = wordStart + fixed.length + 1;
                            editor.setSelectionRange(newPos, newPos);

                            if (fixed !== targetWord) {
                                fixNotice.innerText = `✨ Düzeltildi: "${targetWord}" ➔ "${fixed}"`;
                                setTimeout(() => { fixNotice.innerText = ''; }, 2500);
                            }
                            updateCandidates();
                        })
                        .catch(() => {
                            // Hata durumunda en azından boşluğu koy
                            editor.value = before + ' ' + after;
                            editor.setSelectionRange(cursorPos + 1, cursorPos + 1);
                        });
                    }
                }
            }
        });
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860)
