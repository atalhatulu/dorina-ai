"""
Dorina TTS - Genişletilmiş Veri Hasatçısı (Extended Harvester)
falan42/Turkish-Bentropi-tts-dataset-part-2 ... part-11 depolarındaki
binlerce ek kadın sesi parçasını çeker ve havuza ekler.
"""
import os
import io
import urllib.request
import json
import pandas as pd
import soundfile as sf
import torch
import torchaudio


def harvest_all_bentropi_repos(target_pool_dir: str = "voice/dataset_pool"):
    os.makedirs(target_pool_dir, exist_ok=True)
    wavs_dir = os.path.join(target_pool_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    meta_path = os.path.join(target_pool_dir, "metadata_pool.csv")
    
    # Mevcut dosya sayısını bul
    existing = len([f for f in os.listdir(wavs_dir) if f.endswith(".wav")])
    print(f"[*] Havuzda şu an {existing} ses var. Ek depolar taranıyor...")
    
    parts = [f"falan42/Turkish-Bentropi-tts-dataset-part-{i}" for i in range(2, 12)]
    target_sr = 22050
    total_added = 0
    
    with open(meta_path, "a", encoding="utf-8") as meta_file:
        for repo in parts:
            api_url = f"https://huggingface.co/api/datasets/{repo}"
            req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
            try:
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode())
                    parquets = [s['rfilename'] for s in data.get('siblings', []) if s['rfilename'].endswith('.parquet')]
            except Exception as e:
                print(f"[!] Repo bilgisi alınamadı ({repo}):", e)
                continue
                
            for p_file in parquets:
                p_url = f"https://huggingface.co/datasets/{repo}/resolve/main/{p_file}"
                print(f"[*] İndiriliyor: {repo} -> {p_file}")
                try:
                    df = pd.read_parquet(p_url)
                except Exception as e:
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
                            
                        filename = f"dorina_ext_{existing + total_added:05d}.wav"
                        filepath = os.path.join(wavs_dir, filename)
                        sf.write(filepath, wav_tensor.squeeze(0).numpy(), samplerate=target_sr)
                        
                        meta_file.write(f"{filename}|{text}\n")
                        meta_file.flush()
                        total_added += 1
                        
                        if total_added % 100 == 0:
                            print(f"[+] Havuz Büyüyor: +{total_added} yeni ses eklendi (Toplam: {existing + total_added})")
                    except Exception:
                        continue
                        
    print(f"[TAMAMLANDI] Toplam {total_added} yeni ses havuza başarıyla katıldı!")


if __name__ == "__main__":
    harvest_all_bentropi_repos()
