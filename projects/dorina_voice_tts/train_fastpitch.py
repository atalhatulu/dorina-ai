"""
Dorina TTS - FastPitch Akustik Model Eğitimi (Lokal & Kaggle)
Türkçe karakterleri ve süreleri (Duration) sıfırdan öğrenir.
30.000 adımlık HiFi-GAN vokoderimizin Mel uzayında eğitilir.
"""
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

from voice.text_normalizer import text_to_sequence, VOCAB
from voice.mel_config import MelSpectrogramExtractor
from voice.dorina_fastpitch import DorinaFastPitch


class FastPitchDataset(Dataset):
    def __init__(self, metadata_path: str = "voice/dataset_pool/metadata_pool.csv", wavs_dir: str = "voice/dataset_pool/wavs", max_samples: int = None):
        self.samples = []
        self.wavs_dir = wavs_dir
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "|" in line:
                        fn, text = line.strip().split("|", 1)
                        wp = os.path.join(wavs_dir, fn)
                        if os.path.exists(wp) and len(text) > 5:
                            self.samples.append((wp, text))
                            
        if max_samples:
            self.samples = self.samples[:max_samples]
            
        print(f"[*] FastPitch Veri Seti Yüklendi: Toplam {len(self.samples)} metin-ses çifti.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wp, text = self.samples[idx]
        seq = torch.tensor(text_to_sequence(text), dtype=torch.long)
        wav_np, sr = sf.read(wp)
        wav_tensor = torch.tensor(wav_np, dtype=torch.float32)
        if wav_tensor.dim() > 1:
            wav_tensor = wav_tensor.t()[:1]
        else:
            wav_tensor = wav_tensor.unsqueeze(0)
            
        return {"seq": seq, "wav": wav_tensor.squeeze(0)}


def collate_fastpitch(batch):
    max_seq = max(len(b["seq"]) for b in batch)
    max_wav = max(len(b["wav"]) for b in batch)
    
    padded_seqs = [F.pad(b["seq"], (0, max_seq - len(b["seq"])), value=0) for b in batch]
    padded_wavs = [F.pad(b["wav"], (0, max_wav - len(b["wav"])), value=0.0) for b in batch]
    
    return {
        "seq": torch.stack(padded_seqs),
        "wav": torch.stack(padded_wavs)
    }


def train_fastpitch(total_steps: int = 20000, batch_size: int = 8, lr: float = 1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Dorina FastPitch Eğitimi Başlatılıyor... Cihaz: {device}")
    
    mel_extractor = MelSpectrogramExtractor().to(device)
    model = DorinaFastPitch(vocab_size=len(VOCAB), d_model=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    ds = FastPitchDataset()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fastpitch)
    
    os.makedirs("models", exist_ok=True)
    step = 0
    model.train()
    
    print(f"[*] Eğitim Döngüsü Başladı (Toplam Hedef: {total_steps} Adım)...")
    
    while step < total_steps:
        for b in loader:
            step += 1
            seq = b["seq"].to(device)
            wav = b["wav"].to(device)
            
            with torch.no_grad():
                target_mel = mel_extractor(wav)
                
            optimizer.zero_grad()
            mel_pred, dur_pred, dur_target, pitch_pred, energy_pred = model(seq, target_mel)
            
            # Mel L1 Loss
            min_f = min(mel_pred.shape[-1], target_mel.shape[-1])
            loss_mel = F.l1_loss(mel_pred[:, :, :min_f], target_mel[:, :, :min_f])
            
            # Duration Loss (Log uzayında MSE)
            target_dur_log = torch.log(dur_target.detach() + 1.0)
            loss_dur = F.mse_loss(dur_pred, target_dur_log)
            
            loss = loss_mel * 2.0 + loss_dur
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0 or step == 1:
                print(f"Step [{step}/{total_steps}] - Mel Loss: {loss_mel.item():.4f} | Dur Loss: {loss_dur.item():.4f} | Total: {loss.item():.4f}")
                
            if step % 2500 == 0 or step == total_steps:
                ckpt_path = f"models/dorina_fastpitch_step_{step}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"[CHECKPOINT] FastPitch Modeli Kaydedildi: {ckpt_path}")
                
            if step >= total_steps:
                break

    print("[TAMAMLANDI] FastPitch eğitimi başarıyla tamamlandı!")


if __name__ == "__main__":
    train_fastpitch(total_steps=5000, batch_size=8)
