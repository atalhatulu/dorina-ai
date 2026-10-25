"""
Dorina TTS - Gerçek Türkçe Kadın Sesi Veri İndirici (Parquet Streaming)
"""
import os
import io
import urllib.request
import pandas as pd
import soundfile as sf
import torch
import torchaudio


def download_first_parquet_samples(num_samples: int = 50, output_dir: str = "voice/dataset"):
    os.makedirs(output_dir, exist_ok=True)
    wavs_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.csv")
    
    url = "https://huggingface.co/datasets/falan42/Turkish-Bentropi-tts-dataset/resolve/main/data/train-00000-of-00001.parquet"
    print(f"[*] Gerçek Türkçe TTS verisi taranıyor: {url}")
    
    df = pd.read_parquet(url)
    print(f"[*] Parquet yüklendi! Sütunlar: {df.columns.tolist()} | Toplam satır: {len(df)}")
    
    metadata_lines = []
    target_sr = 22050
    saved = 0
    
    for idx, row in df.iterrows():
        if saved >= num_samples:
            break
            
        audio_info = row.get("audio")
        text = str(row.get("transcription") or row.get("text") or "").strip()
        
        if not text or len(text) < 5 or not audio_info:
            continue
            
        audio_bytes = audio_info.get("bytes")
        if not audio_bytes:
            continue
            
        # Byte'tan ses dalgasını oku
        wav_data, sr = sf.read(io.BytesIO(audio_bytes))
        wav_tensor = torch.tensor(wav_data, dtype=torch.float32)
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        else:
            wav_tensor = wav_tensor.t()[:1]  # Mono yap
            
        # 22050 Hz'e dönüştür
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            wav_tensor = resampler(wav_tensor)
            
        filename = f"dorina_tr_{saved:04d}.wav"
        filepath = os.path.join(wavs_dir, filename)
        sf.write(filepath, wav_tensor.squeeze(0).numpy(), samplerate=target_sr)
        
        metadata_lines.append(f"{filename}|{text}")
        saved += 1
        
        if saved % 10 == 0:
            print(f"[+] {saved}/{num_samples} gerçek ses WAV olarak kaydedildi...")

    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))
        
    print(f"[OK] {saved} adet gerçek Türkçe ses {wavs_dir} klasörüne hazırlandı!")


if __name__ == "__main__":
    download_first_parquet_samples(num_samples=30)
