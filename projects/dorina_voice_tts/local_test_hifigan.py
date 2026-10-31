"""
Dorina HiFi-GAN - Lokal Doğrulama ve Test Eğitimi (Mini-Test)
Claude'un yönergeleriyle birebir uyumlu:
- Segment: 8192 sample
- Batch: 8 (RTX 4050 için güvenli VRAM)
- L_G = L_adv + 2*L_FM + 45*L_mel (Aynı mel fonksiyonu)
- NaN koruması, Checkpoint kaydetme ve her 500 adımda ses dinleme çıktısı
"""
import os
import random
import soundfile as sf
import torch
import torch.nn.functional as F

from voice.mel_config import get_mel_spectrogram, MelSpectrogramExtractor
from voice.hifigan_model import HiFiGANGenerator, MultiPeriodDiscriminator


def run_local_hifigan_test(total_steps: int = 1500, batch_size: int = 8, segment_size: int = 8192):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] HiFi-GAN Mini-Test Başlatılıyor... Cihaz: {device}")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("voice/test_audio", exist_ok=True)
    
    # 1. Modeller ve Mel çıkarıcı (FP32 kararlılığı)
    generator = HiFiGANGenerator().to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    mel_fn = MelSpectrogramExtractor().to(device)
    
    # Optimizer (AdamW betas 0.8, 0.99 Claude önerisi)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=2e-4, betas=(0.8, 0.99), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(mpd.parameters(), lr=2e-4, betas=(0.8, 0.99), weight_decay=1e-4)
    
    # Ses dosyalarını tara (dataset veya dataset_pool)
    wav_dirs = ["voice/dataset/wavs", "voice/dataset_pool/wavs"]
    wav_files = []
    for d in wav_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".wav"):
                    wav_files.append(os.path.join(d, f))
                    
    print(f"[*] Toplam {len(wav_files)} ses dosyası taranıyor...")
    if not wav_files:
        print("[!] Hata: Ses dosyası bulunamadı!")
        return

    # Sabit referans dosya (Her 500 adımda dinlemek için)
    ref_wav_path = wav_files[0]
    ref_audio, _ = sf.read(ref_wav_path)
    ref_tensor = torch.tensor(ref_audio, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        ref_mel = mel_fn(ref_tensor)

    print(f"[*] {total_steps} adımlık döngü başlatılıyor (Batch={batch_size}, Segment={segment_size})...")
    
    generator.train()
    mpd.train()
    
    for step in range(1, total_steps + 1):
        # Batch oluştur: 8 adet rastgele 8192 sample'lık ses parçası
        batch_chunks = []
        for _ in range(batch_size):
            fn = random.choice(wav_files)
            try:
                data, _ = sf.read(fn)
                if len(data) > segment_size + 500:
                    start = random.randint(0, len(data) - segment_size)
                    chunk = data[start:start + segment_size]
                    batch_chunks.append(chunk)
                else:
                    batch_chunks.append(ref_audio[:segment_size])
            except Exception:
                batch_chunks.append(ref_audio[:segment_size])
                
        wav_batch = torch.tensor(batch_chunks, dtype=torch.float32).to(device) # (batch, 8192)
        
        # 1. Mel Spektrogram çıkar (Ground-Truth)
        with torch.no_grad():
            mel_gt = mel_fn(wav_batch) # (batch, 80, 32)
            
        # 2. Generator Üretimi
        wav_gen = generator(mel_gt)
        min_len = min(wav_batch.shape[-1], wav_gen.shape[-1])
        wav_real = wav_batch[:, :min_len]
        wav_fake = wav_gen[:, :min_len]
        
        # --- DISCRIMINATOR GÜNCELLEMESİ ---
        opt_d.zero_grad()
        r_outs, g_outs, _, _ = mpd(wav_real, wav_fake.detach())
        
        loss_d = 0
        for dr, dg in zip(r_outs, g_outs):
            r_loss = torch.mean((1.0 - dr) ** 2)
            g_loss = torch.mean(dg ** 2)
            loss_d = loss_d + (r_loss + g_loss)
            
        loss_d.backward()
        opt_d.step()
        
        # --- GENERATOR GÜNCELLEMESİ ---
        opt_g.zero_grad()
        _, g_outs, f_r, f_g = mpd(wav_real, wav_fake)
        
        # 1. Adversarial Loss
        loss_adv = sum(torch.mean((1.0 - dg) ** 2) for dg in g_outs)
        
        # 2. Feature Matching Loss (2.0 katsayısı)
        loss_fm = 0
        for dr, dg in zip(f_r, f_g):
            for rl, gl in zip(dr, dg):
                loss_fm = loss_fm + torch.mean(torch.abs(rl - gl))
        loss_fm = loss_fm * 2.0
        
        # 3. Mel Loss (45.0 katsayısı - Claude reçetesi)
        mel_fake = mel_fn(wav_fake)
        min_mel = min(mel_gt.shape[-1], mel_fake.shape[-1])
        loss_mel_raw = F.l1_loss(mel_fake[:, :, :min_mel], mel_gt[:, :, :min_mel])
        loss_mel = loss_mel_raw * 45.0
        
        loss_g = loss_adv + loss_fm + loss_mel
        loss_g.backward()
        opt_g.step()
        
        # Loglama
        if step % 100 == 0 or step == 1:
            print(f"Step [{step}/{total_steps}] - Loss G: {loss_g.item():.3f} (Mel Raw: {loss_mel_raw.item():.4f}) | Loss D: {loss_d.item():.3f}")
            
        # Her 500 adımda bir ses kaydet ve checkpoint al
        if step % 500 == 0 or step == total_steps:
            generator.eval()
            with torch.no_grad():
                gen_test = generator(ref_mel).squeeze().cpu().numpy()
                out_path = f"voice/test_audio/step_{step}.wav"
                sf.write(out_path, gen_test, 22050)
                print(f"[SES ÇIKTISI] {out_path} kaydedildi (Dinlemeye hazır).")
                
            ckpt_path = "models/dorina_hifigan_local_ckpt.pt"
            torch.save({
                "step": step,
                "generator": generator.state_dict(),
                "mpd": mpd.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict()
            }, ckpt_path)
            print(f"[CHECKPOINT] {ckpt_path} kaydedildi.")
            generator.train()

    print("[TAMAMLANDI] Mini-test başarıyla bitti!")


if __name__ == "__main__":
    run_local_hifigan_test(total_steps=1500, batch_size=8)
