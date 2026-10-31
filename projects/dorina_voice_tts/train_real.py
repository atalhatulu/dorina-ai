"""
Dorina TTS - Gerçek Ses Verisi ile Eğitim Döngüsü (Lokal & Sunucu)
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from voice.text_processor import VOCAB
from voice.audio_dsp import AudioProcessor
from voice.dorina_tts_model import DorinaTTSModel
from voice.dataset_loader import DorinaVoiceDataset, collate_voice_batch


def train_real_dorina(epochs: int = 50, batch_size: int = 4, lr: float = 5e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Gerçek ses eğitimi başlatılıyor... Cihaz: {device}")
    
    dataset = DorinaVoiceDataset("voice/dataset")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_voice_batch)
    
    model = DorinaTTSModel(vocab_size=len(VOCAB)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_mel = nn.L1Loss()
    criterion_var = nn.MSELoss()
    
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for b_idx, batch in enumerate(dataloader):
            seq = batch["seq"].to(device)
            target_mel = batch["mel"].to(device)
            dur = batch["dur"].to(device)
            pitch = batch["pitch"].to(device)
            energy = batch["energy"].to(device)
            
            optimizer.zero_grad()
            pred_mel, pred_dur, pred_pitch, pred_energy = model(seq, dur, pitch, energy)
            
            # Mel loss
            min_time = min(pred_mel.shape[-1], target_mel.shape[-1])
            loss_mel = criterion_mel(pred_mel[:, :, :min_time], target_mel[:, :, :min_time])
            loss_dur = criterion_var(pred_dur, dur)
            loss_pitch = criterion_var(pred_pitch, pitch)
            loss_energy = criterion_var(pred_energy, energy)
            
            loss = loss_mel * 2.0 + loss_dur + loss_pitch + loss_energy
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f} (Mel: {loss_mel.item():.4f})")
            
    os.makedirs("models", exist_ok=True)
    save_path = "models/dorina_tts_real.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[OK] Gerçek kadın sesi modeli kaydedildi: {save_path}")


if __name__ == "__main__":
    train_real_dorina(epochs=250, batch_size=4, lr=5e-4)
