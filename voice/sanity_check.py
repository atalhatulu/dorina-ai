"""
Dorina TTS - Sanity Test Eğitimi (Tek Cümleye Aşırı Uyum / Overfit)
Claude'un 4. maddesi: Modeli tüm veriyle eğitmeye başlamadan önce tek bir ses kaydına
overfit edip harfleri ve sesi kristal netliğinde çıkartabildiğini kanıtlar.
"""
import soundfile as sf
import torch
import torch.nn.functional as F

from voice.text_normalizer import text_to_sequence, VOCAB
from voice.audio_dsp import AudioProcessor
from voice.dorina_fastpitch import DorinaFastPitch


def run_sanity_check(epochs: int = 150):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Sanity Check başlatılıyor (Cihaz: {device})...")
    
    dsp = AudioProcessor().to(device)
    model = DorinaFastPitch(vocab_size=len(VOCAB), d_model=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Gerçek eğitim verisindeki ilk ses ve transkripti
    wav_path = "voice/dataset/wavs/dorina_tr_0002.wav"
    text = "Ward yayınladığı kitapçıkta ya da broşürde olayın detaylarını şöyle anlatıyordu."
    
    wav_np, sr = sf.read(wav_path)
    wav_tensor = torch.tensor(wav_np, dtype=torch.float32).to(device)
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
        
    with torch.no_grad():
        target_mel = dsp.wav_to_mel(wav_tensor) # (1, 80, frames)
        
    seq = torch.tensor([text_to_sequence(text)], dtype=torch.long, device=device)
    
    print(f"[*] Hedef Cümle: '{text}'")
    print(f"[*] Harf Sayısı: {seq.shape[-1]} | Ses Frame Sayısı: {target_mel.shape[-1]}")
    
    model.train()
    for ep in range(1, epochs + 1):
        optimizer.zero_grad()
        mel_out, dur_pred, dur_target, pitch_pred, energy_pred = model(seq, target_mel)
        
        # 1. Mel Loss
        min_f = min(mel_out.shape[-1], target_mel.shape[-1])
        loss_mel = F.l1_loss(mel_out[:, :, :min_f], target_mel[:, :, :min_f])
        
        # 2. Duration Loss (log space)
        target_dur_log = torch.log(dur_target + 1.0)
        loss_dur = F.mse_loss(dur_pred, target_dur_log)
        
        loss = loss_mel * 2.0 + loss_dur
        loss.backward()
        optimizer.step()
        
        if ep % 25 == 0 or ep == 1:
            print(f"Epoch [{ep}/{epochs}] - Total Loss: {loss.item():.4f} (Mel: {loss_mel.item():.4f}, Dur: {loss_dur.item():.4f})")
            
    # Eğitilen tek cümlelik modelden inferans alalım
    model.eval()
    with torch.no_grad():
        mel_gen, _, _, _, _ = model(seq)
        rec_wav = dsp.mel_to_wav(mel_gen.squeeze(0)).cpu().numpy()
        sf.write("sanity_check_output.wav", rec_wav, 22050)
        
    print("[OK] Sanity Check tamamlandı! Çıktı: sanity_check_output.wav")


if __name__ == "__main__":
    run_sanity_check(epochs=150)
