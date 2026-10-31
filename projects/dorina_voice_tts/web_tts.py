"""
Dorina TTS - Web Doğrulama ve Dinleme Arayüzü
Kullanıcının beğendiği gerçek kadın sesini ve eğitilen vokoderin gelişimini (step_500, step_1000, step_1500)
doğrudan tarayıcıda bellekten çalar.
"""
import io
import os
import soundfile as sf
import torch
from fastapi import FastAPI, Request
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel
import uvicorn

from voice.mel_config import get_mel_spectrogram
from voice.hifigan_model import HiFiGANGenerator

app = FastAPI(title="Dorina TTS Studio")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Eğitilen Vokoder
vocoder = HiFiGANGenerator().to(device)
ckpt_path = "models/dorina_hifigan_local_ckpt.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "generator" in ckpt:
        vocoder.load_state_dict(ckpt["generator"])
    else:
        vocoder.load_state_dict(ckpt)
vocoder.eval()

# 2. Beğenilen Orijinal Kadın Sesi (Hafızada tutulur)
CANDIDATE_PATH = "test_female_candidate.wav"


@app.get("/", response_class=HTMLResponse)
def index():
    html = """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <title>Dorina TTS - Canlı Dinleme Stüdyosu</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, sans-serif; }
            body { background: #0f172a; color: #f8fafc; display: flex; align-items: center; justify-content: center; min-height: 100vh; padding: 20px; }
            .card { background: #1e293b; border: 1px solid #334155; border-radius: 16px; padding: 32px; width: 100%; max-width: 650px; }
            h1 { font-size: 22px; color: #38bdf8; margin-bottom: 8px; }
            p { font-size: 14px; color: #94a3b8; margin-bottom: 20px; }
            .section { background: #0f172a; border: 1px solid #334155; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
            .section-title { font-size: 15px; font-weight: 600; color: #f1f5f9; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
            .tag { font-size: 11px; background: #0284c7; color: white; padding: 3px 8px; border-radius: 6px; }
            audio { width: 100%; margin-top: 8px; outline: none; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🎙️ Dorina Ses Laboratuvarı</h1>
            <p>Demin dinlediğin ve kelimeleri seçilen o gerçek kadın sesi ve vokoderin ürettiği sesler aşağıda hazır:</p>

            <div class="section">
                <div class="section-title">
                    <span>1. Beğendiğin Orijinal Kadın Konuşmacı</span>
                    <span class="tag">Hedef Dorina Sesi</span>
                </div>
                <div style="font-size: 13px; color: #cbd5e1; margin-bottom: 6px;">
                    <i>"Her geçen gün büyüyen LGBT sapkınlığına Instagram'dan bir destek daha geldi..."</i>
                </div>
                <audio controls src="/stream/target_female"></audio>
            </div>

            <div class="section">
                <div class="section-title">
                    <span>2. Bizim Modelin Sıfırdan Eğitilmiş Çıktısı (Step 1500)</span>
                    <span class="tag">HiFi-GAN 1.500 Adım</span>
                </div>
                <div style="font-size: 13px; color: #cbd5e1; margin-bottom: 6px;">
                    <i>"1885'te James B. Ward adlı bir yazar, uzun süredir kayıp olan bir hazine..."</i>
                </div>
                <audio controls src="/stream/our_model_1500"></audio>
            </div>

            <div class="section">
                <div class="section-title">
                    <span>3. Vokoder Canlı Rekonstrüksiyonu</span>
                    <span class="tag">Canlı Mel -> Vokoder</span>
                </div>
                <div style="font-size: 13px; color: #cbd5e1; margin-bottom: 6px;">
                    Hedef kadın sesinin Mel'i çıkarılıp bizim vokoder tarafından anlık üretiliyor:
                </div>
                <audio controls src="/stream/live_reconstruction"></audio>
            </div>
        </div>
    </body>
    </html>
    """
    return html


@app.get("/stream/target_female")
def stream_target_female():
    if os.path.exists(CANDIDATE_PATH):
        data, sr = sf.read(CANDIDATE_PATH)
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")
    return Response(status_code=404)


@app.get("/stream/our_model_1500")
def stream_model_1500():
    p = "voice/test_audio/step_1500.wav"
    if os.path.exists(p):
        data, sr = sf.read(p)
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")
    return Response(status_code=404)


@app.get("/stream/live_reconstruction")
def stream_live_reconstruction():
    if os.path.exists(CANDIDATE_PATH):
        data, sr = sf.read(CANDIDATE_PATH)
        t = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
        mel = get_mel_spectrogram(t)
        with torch.no_grad():
            gen_wav = vocoder(mel).squeeze().cpu().numpy()
        buf = io.BytesIO()
        sf.write(buf, gen_wav, 22050, format="WAV")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")
    return Response(status_code=404)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860)
