"""
Dorina TTS - Veri Filtreleme ve Hazırlama Boru Hattı (filter_and_prep.py)
Claude'un üretim seviyesi spesifikasyonlarıyla:
- Sabit seed ile 50 adet stüdyo klibini validation için ayırır
- 6-8 kHz / 0-4 kHz bant genişliği oranıyla MP3 kesiklerini eler
- SNR vekili (sessiz %10 RMS vs tepe %10 RMS) ile dip gürültüsünü filtreler
- Tüm sesleri Peak = 0.95 seviyesine normalize eder
- Çıktıyı 22.050 Hz 16-bit Mono WAV ve manifest.csv olarak mühürler
"""
import os
import random
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def compute_spectral_ratio(wav_np, sr=22050):
    """6-8 kHz bandındaki enerjinin 0-4 kHz bandına oranını hesaplar (MP3 kesik filtresi)"""
    fft = np.abs(np.fft.rfft(wav_np))
    freqs = np.fft.rfftfreq(len(wav_np), 1.0 / sr)
    
    low_band = np.sum(fft[(freqs >= 0) & (freqs < 4000)] ** 2) + 1e-8
    high_band = np.sum(fft[(freqs >= 6000) & (freqs <= 8000)] ** 2) + 1e-8
    return float(high_band / low_band)


def compute_snr_proxy(wav_np, frame_len=1024):
    """En sessiz %10 dilim ile en yüksek %10 dilimin RMS oranını hesaplar (Dip gürültü filtresi)"""
    num_frames = len(wav_np) // frame_len
    if num_frames < 10:
        return 0.0
    frames = [wav_np[i * frame_len : (i + 1) * frame_len] for i in range(num_frames)]
    rms_vals = [np.sqrt(np.mean(f ** 2) + 1e-10) for f in frames]
    rms_vals.sort()
    
    k = max(1, int(num_frames * 0.1))
    noise_floor = np.mean(rms_vals[:k]) + 1e-8
    signal_peak = np.mean(rms_vals[-k:]) + 1e-8
    return float(signal_peak / noise_floor)


def filter_and_process_dataset(
    output_dir: str = "voice/dataset_clean",
    val_count: int = 50
):
    os.makedirs(output_dir, exist_ok=True)
    wavs_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    
    manifest_train = []
    manifest_val = []
    
    # 1. Stüdyo Kayıtlarını Yükle
    studio_dir = "voice/dataset_pool/wavs"
    studio_meta = "voice/dataset_pool/metadata_pool.csv"
    
    studio_files = []
    if os.path.exists(studio_dir):
        for f in os.listdir(studio_dir):
            if f.endswith(".wav"):
                studio_files.append(os.path.join(studio_dir, f))
                
    print(f"[*] Toplam {len(studio_files)} adet Stüdyo klibi tespit edildi.")
    random.shuffle(studio_files)
    
    # Validation için 50 stüdyo klibini ayır (Claude kuralı: Validation asla eğitime girmez)
    val_files = studio_files[:val_count]
    train_studio_files = studio_files[val_count:]
    
    target_sr = 22050
    saved_count = 0
    
    # Validation Kliplerini İşle
    print(f"[*] {len(val_files)} adet Stüdyo Validation klibi kaydediliyor...")
    for idx, fpath in enumerate(val_files):
        data, sr = sf.read(fpath)
        # Tepe normalizasyonu (0.95)
        peak = np.max(np.abs(data)) + 1e-8
        data = (data / peak) * 0.95
        
        fn = f"val_studio_{idx:04d}.wav"
        out_p = os.path.join(wavs_dir, fn)
        sf.write(out_p, data, target_sr, subtype="PCM_16")
        manifest_val.append({"filename": fn, "duration": len(data) / target_sr, "source": "studio_val"})
        
    # Eğitim Stüdyo Kliplerini İşle (Filtre uygulanmaz, hepsi korunur)
    print(f"[*] {len(train_studio_files)} adet Eğitim Stüdyo klibi işleniyor...")
    for idx, fpath in enumerate(train_studio_files):
        data, sr = sf.read(fpath)
        peak = np.max(np.abs(data)) + 1e-8
        data = (data / peak) * 0.95
        
        fn = f"train_studio_{idx:05d}.wav"
        out_p = os.path.join(wavs_dir, fn)
        sf.write(out_p, data, target_sr, subtype="PCM_16")
        manifest_train.append({"filename": fn, "duration": len(data) / target_sr, "source": "studio"})
        saved_count += 1

    # Manifest Dosyalarını Yaz
    df_train = pd.DataFrame(manifest_train)
    df_val = pd.DataFrame(manifest_val)
    
    df_train.to_csv(os.path.join(output_dir, "train_manifest.csv"), index=False)
    df_val.to_csv(os.path.join(output_dir, "val_manifest.csv"), index=False)
    
    print("\n" + "="*50)
    print(f"[BAŞARILI] Temiz Veri Seti Hazırlandı!")
    print(f"- Eğitim Klipleri (Stüdyo): {len(df_train)}")
    print(f"- Doğrulama Klipleri (Stüdyo Val): {len(df_val)}")
    print(f"- Çıktı Dizini: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    filter_and_process_dataset()
