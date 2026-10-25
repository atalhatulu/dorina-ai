"""
Dorina TTS - Kaggle & Lokal Eğitim Pipeline Scripti
Tek tıkla çalıştırılabilir, veriyi otomatik hazırlar ve sıfırdan modeli eğitir.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from voice.text_processor import text_to_sequence, VOCAB
from voice.audio_dsp import AudioProcessor, AudioConfig
from voice.dorina_tts_model import DorinaTTSModel


class DorinaSampleDataset(Dataset):
    """
    Dorina Mini Prototip Veri Kümesi
    """
    def __init__(self, data_samples: list[dict]):
        self.samples = data_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        seq = torch.tensor(text_to_sequence(item["text"]), dtype=torch.long)
        mel = item["mel"]
        seq_len = len(seq)
        mel_len = mel.shape[-1]
        
        # Ortalama harf süresi (basit hizalama başlangıcı)
        dur = torch.full((seq_len,), fill_value=max(1, mel_len // seq_len), dtype=torch.float)
        # Kalan frame farkını son harfe ekle
        diff = mel_len - int(dur.sum().item())
        dur[-1] = max(1, dur[-1] + diff)
        
        pitch = torch.randn(seq_len)
        energy = torch.randn(seq_len)
        
        return {
            "seq": seq,
            "mel": mel,
            "dur": dur,
            "pitch": pitch,
            "energy": energy
        }


def collate_fn(batch):
    # Padding işlemi
    max_seq_len = max(len(b["seq"]) for b in batch)
    max_mel_len = max(b["mel"].shape[-1] for b in batch)
    
    padded_seqs, padded_mels, padded_durs, padded_pitches, padded_energies = [], [], [], [], []
    
    for b in batch:
        # Pad seq
        pad_s = max_seq_len - len(b["seq"])
        padded_seqs.append(torch.nn.functional.pad(b["seq"], (0, pad_s), value=0))
        padded_durs.append(torch.nn.functional.pad(b["dur"], (0, pad_s), value=0))
        padded_pitches.append(torch.nn.functional.pad(b["pitch"], (0, pad_s), value=0))
        padded_energies.append(torch.nn.functional.pad(b["energy"], (0, pad_s), value=0))
        
        # Pad mel
        pad_m = max_mel_len - b["mel"].shape[-1]
        padded_mels.append(torch.nn.functional.pad(b["mel"], (0, pad_m), value=-11.5))
        
    return {
        "seq": torch.stack(padded_seqs),
        "mel": torch.stack(padded_mels),
        "dur": torch.stack(padded_durs),
        "pitch": torch.stack(padded_pitches),
        "energy": torch.stack(padded_energies)
    }


def train_dorina_tts(epochs: int = 100, lr: float = 1e-3, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Dorina TTS Eğitimi Başlatılıyor... Cihaz: {device}")
    
    dsp = AudioProcessor().to(device)
    model = DorinaTTSModel(vocab_size=len(VOCAB)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_mel = nn.L1Loss()
    criterion_var = nn.MSELoss()
    
    # Hızlı Doğrulama için Sentetik Örnek Cümleler
    training_texts = [
        "merhaba ben dorina.",
        "senin yapay zeka asistanınım.",
        "bugün sana nasıl yardımcı olabilirim?",
        "kod yazmayı ve öğrenmeyi çok seviyorum.",
        "talha ile birlikte çalışmak harika."
    ]
    
    samples = []
    for t in training_texts:
        # Test için 22.050Hz temiz sentetik ses dalgası simülasyonu
        duration_sec = max(1.0, len(t) * 0.08)
        num_samples = int(duration_sec * 22050)
        simulated_wav = torch.sin(torch.linspace(0, 100, num_samples)).unsqueeze(0).to(device)
        mel = dsp.wav_to_mel(simulated_wav).squeeze(0).cpu()
        samples.append({"text": t, "mel": mel})
        
    dataset = DorinaSampleDataset(samples)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    print(f"[*] Toplam {len(samples)} örnek ile döngü başlatılıyor...")
    model.train()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in dataloader:
            seq = batch["seq"].to(device)
            target_mel = batch["mel"].to(device)
            dur = batch["dur"].to(device)
            pitch = batch["pitch"].to(device)
            energy = batch["energy"].to(device)
            
            optimizer.zero_grad()
            pred_mel, pred_dur, pred_pitch, pred_energy = model(seq, dur, pitch, energy)
            
            # Mel loss
            # Zaman boyutu padding farkını hizala
            min_time = min(pred_mel.shape[-1], target_mel.shape[-1])
            loss_mel = criterion_mel(pred_mel[:, :, :min_time], target_mel[:, :, :min_time])
            loss_dur = criterion_var(pred_dur, dur)
            loss_pitch = criterion_var(pred_pitch, pitch)
            loss_energy = criterion_var(pred_energy, energy)
            
            loss = loss_mel * 2.0 + loss_dur + loss_pitch + loss_energy
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {total_loss / len(dataloader):.4f}")
            
    os.makedirs("models", exist_ok=True)
    save_path = "models/dorina_tts_prototype.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[OK] Model başarıyla eğitildi ve kaydedildi: {save_path}")


if __name__ == "__main__":
    train_dorina_tts(epochs=20)
