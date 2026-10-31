import os, sys, random, torch
import soundfile as sf
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))

from voice.dorina_fastpitch import DorinaFastPitch
from voice.mel_config import MelSpectrogramExtractor
from voice.text_normalizer import text_to_sequence

print("[*] Localde mimari, veri akisi ve egitim adimlari dogrulaniyor...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cihaz: {device}")

mel_extractor = MelSpectrogramExtractor().to(device)
model = DorinaFastPitch().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

wav_dir = "voice/dataset_clean/wavs"
wav_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith(".wav")]
print(f"Bulunan temiz WAV sayisi: {len(wav_files)}")

# 5 adim gercek egitim adimi calistir
model.train()
for step in range(1, 6):
    batch_files = random.sample(wav_files, 4)
    texts = [
        "merhaba dorina ben talha",
        "bugün hava çok güzel",
        "yapay zeka ses modeli eğitiliyor",
        "yeni ses sentezleme testi"
    ]
    
    seqs = [torch.tensor(text_to_sequence(t), dtype=torch.long) for t in texts]
    wavs = []
    for f in batch_files:
        w, sr = sf.read(f)
        if len(w) > 22050 * 3:
            w = w[:22050 * 3]
        wavs.append(torch.tensor(w, dtype=torch.float32))
        
    max_s = max(len(s) for s in seqs)
    max_w = max(len(w) for w in wavs)
    
    seq_b = torch.stack([F.pad(s, (0, max_s - len(s))) for s in seqs]).to(device)
    wav_b = torch.stack([F.pad(w, (0, max_w - len(w))) for w in wavs]).to(device)
    
    with torch.no_grad():
        target_mel = mel_extractor(wav_b)
        
    optimizer.zero_grad()
    mel_pred, d_pred, d_target, p_pred, e_pred = model(seq_b, target_mel)
    
    min_f = min(mel_pred.shape[-1], target_mel.shape[-1])
    loss_mel = F.l1_loss(mel_pred[:, :, :min_f], target_mel[:, :, :min_f])
    loss_dur = F.mse_loss(d_pred, torch.log(d_target.detach() + 1.0))
    loss = loss_mel * 2.0 + loss_dur
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
        
    print(f"Step {step}/5 Basarili! Mel Loss: {loss_mel.item():.4f} | Dur Loss: {loss_dur.item():.4f}")

# Inference (Sentez) Dogrulamasi
model.eval()
with torch.no_grad():
    test_seq = torch.tensor(text_to_sequence("merhaba dorina")).unsqueeze(0).to(device)
    mel_out, _, _, _, _ = model(test_seq)
    print(f"Inference Basarili! Uretilen Mel Spectrogram Sekli: {mel_out.shape}")

print("[TAMAM] Local dogrulama SIFIR HATA ile tamamlandi!")
