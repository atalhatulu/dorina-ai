"""
Dorina TTS - Toplu Veri İndirici (Background Dataset Harvester)
Mevcut eğitim devam ederken arka planda diğer parçalardaki yüzlerce stüdyo kaydı kadın sesini çeker
ve 'voice/dataset_pool/' klasörüne depolar.
"""
import os
import io
import pandas as pd
import soundfile as sf
import torch
import torchaudio


def harvest_all_bentropi(target_pool_dir: str = "voice/dataset_pool"):
    os.makedirs(target_pool_dir, exist_ok=True)
    wavs_dir = os.path.join(target_pool_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    meta_path = os.path.join(target_pool_dir, "metadata_pool.csv")
    
    # 2'den 11'e kadar tüm diğer parquet dosyalarını tara
    parquet_files = [
        "data/train-00000-of-00002.parquet",
        "data/train-00000-of-00003.parquet",
        "data/train-00000-of-00004.parquet",
        "data/train-00000-of-00005.parquet",
        "data/train-00000-of-00006.parquet",
        "data/train-00000-of-00007.parquet",
        "data/train-00000-of-00008.parquet",
        "data/train-00000-of-00009.parquet",
        "data/train-00000-of-000010.parquet",
        "data/train-00000-of-000011.parquet"
    ]
    
    total_saved = 0
    target_sr = 22050
    
    with open(meta_path, "a", encoding="utf-8") as meta_file:
        for p_file in parquet_files:
            url = f"https://huggingface.co/datasets/falan42/Turkish-Bentropi-tts-dataset/resolve/main/{p_file}"
            print(f"[*] Arka plan: Yeni parça taranıyor -> {p_file}")
            try:
                df = pd.read_parquet(url)
            except Exception as e:
                print(f"[!] Hata oluştu ({p_file}):", e)
                continue
                
            for idx, row in df.iterrows():
                audio_info = row.get("audio")
                text = str(row.get("transcription") or row.get("text") or "").strip()
                if not text or len(text) < 5 or not audio_info:
                    continue
                audio_bytes = audio_info.get("bytes")
                if not audio_bytes:
                    continue
                    
                try:
                    wav_data, sr = sf.read(io.BytesIO(audio_bytes))
                    wav_tensor = torch.tensor(wav_data, dtype=torch.float32)
                    if wav_tensor.dim() == 1:
                        wav_tensor = wav_tensor.unsqueeze(0)
                    else:
                        wav_tensor = wav_tensor.t()[:1]
                        
                    if sr != target_sr:
                        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                        wav_tensor = resampler(wav_tensor)
                        
                    filename = f"dorina_pool_{total_saved:05d}.wav"
                    filepath = os.path.join(wavs_dir, filename)
                    sf.write(filepath, wav_tensor.squeeze(0).numpy(), samplerate=target_sr)
                    
                    meta_file.write(f"{filename}|{text}\n")
                    meta_file.flush()
                    total_saved += 1
                    
                    if total_saved % 50 == 0:
                        print(f"[+] Havuz Genişliyor: {total_saved} adet yeni kadın sesi toplandı...")
                except Exception as ex:
                    continue
                    
    print(f"[TAMAMLANDI] Toplam {total_saved} adet yeni kadın sesi başarıyla havuza eklendi!")


if __name__ == "__main__":
    harvest_all_bentropi()
