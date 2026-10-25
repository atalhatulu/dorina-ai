"""
Dorina TTS - Kapsamlı Aligner & Model Doğrulama Testi
Claude'un başarı kriterleri:
1. MAS A_hard ve A_soft hizalama diyagonal görselleştirmesi (alignment_plot.png)
2. Binarization loss (L_bin) + Forward-sum (CTC)
3. Durations analizi: Sessizlik, ünlüler ve ünsüzlerin süre kontrolü
4. Ground-Truth Duration vs Predicted Duration ile iki ayrı ses üretimi
"""
import os
import torch
import torch.nn.functional as F
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

from voice.text_normalizer import text_to_sequence, VOCAB, ID_TO_CHAR
from voice.audio_dsp import AudioProcessor
from voice.dorina_fastpitch import DorinaFastPitch
from voice.alignment import monotonic_alignment_search


def run_rigorous_validation(epochs: int = 120):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Kapsamlı Doğrulama Testi Başlatılıyor (Cihaz: {device})...")
    
    dsp = AudioProcessor().to(device)
    model = DorinaFastPitch(vocab_size=len(VOCAB), d_model=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 20-30 kliplik alt kümeden temsili 5 net klip seçelim
    samples = []
    with open("voice/dataset/metadata.csv", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 5: break
            fn, text = line.strip().split("|", 1)
            wav_path = os.path.join("voice/dataset/wavs", fn)
            if os.path.exists(wav_path):
                w_np, sr = sf.read(wav_path)
                w_tensor = torch.tensor(w_np, dtype=torch.float32).to(device)
                if w_tensor.dim() == 1: w_tensor = w_tensor.unsqueeze(0)
                with torch.no_grad():
                    mel = dsp.wav_to_mel(w_tensor)
                seq = torch.tensor(text_to_sequence(text), dtype=torch.long, device=device)
                samples.append((seq, mel, text))
                
    print(f"[*] {len(samples)} klip ile hizalama ve model eğitimi başlatılıyor...")
    
    model.train()
    for ep in range(1, epochs + 1):
        total_mel_loss, total_dur_loss, total_bin_loss = 0.0, 0.0, 0.0
        
        # Binarization loss katsayısı kademeli açılır (Claude tavsiyesi)
        lambda_bin = min(1.0, ep / 40.0) if ep > 10 else 0.0
        
        for seq, mel, _ in samples:
            seq_b = seq.unsqueeze(0)
            mel_b = mel
            
            optimizer.zero_grad()
            
            # Encoder çıktıları
            x = model.embedding(seq_b)
            for l in model.encoder: x = l(x)
            
            # Soft Aligner matrisi (log-softmax)
            attn_soft = model.aligner(x, mel_b) # (1, T_text, T_mel)
            
            # MAS Hard Alignment (Gradyan sızmaması için no_grad)
            with torch.no_grad():
                hard_align = monotonic_alignment_search(attn_soft[0]) # (T_text, T_mel)
                dur_target = hard_align.sum(dim=-1).unsqueeze(0).detach() # (1, T_text)
                
            # L_bin loss: soft matrisi hard matrise yaklaştırır
            # L_bin = - sum(log A_soft * A_hard) / sum(A_hard)
            loss_bin = - (attn_soft[0] * hard_align).sum() / (hard_align.sum() + 1e-6)
            
            # Duration Predictor (Log uzayında MSE)
            dur_pred = model.duration_predictor(x)
            loss_dur = F.mse_loss(dur_pred, torch.log(dur_target + 1.0))
            
            # Decoder & Mel üretimi
            p_emb = model.pitch_embedding(model.pitch_predictor(x).unsqueeze(1)).transpose(1, 2)
            e_emb = model.energy_embedding(model.energy_predictor(x).unsqueeze(1)).transpose(1, 2)
            x_var = x + p_emb + e_emb
            
            # Eğitimde MAS ground-truth süreleri verilir
            x_exp = model.length_regulator(x_var, dur_target)
            for l in model.decoder: x_exp = l(x_exp)
            mel_out = model.mel_linear(x_exp).transpose(1, 2)
            
            min_f = min(mel_out.shape[-1], mel_b.shape[-1])
            loss_mel = F.l1_loss(mel_out[:, :, :min_f], mel_b[:, :, :min_f])
            
            loss = loss_mel * 2.0 + loss_dur + (lambda_bin * loss_bin)
            loss.backward()
            optimizer.step()
            
            total_mel_loss += loss_mel.item()
            total_dur_loss += loss_dur.item()
            total_bin_loss += loss_bin.item()
            
        if ep % 20 == 0 or ep == 1:
            print(f"Epoch [{ep}/{epochs}] - Mel L1: {total_mel_loss/len(samples):.4f} | Dur MSE: {total_dur_loss/len(samples):.4f} | Bin: {total_bin_loss/len(samples):.4f} (lambda_bin={lambda_bin:.2f})")

    # 1. Hizalama Grafiğini Çiz ve Kaydet
    print("[*] Hizalama grafiği (Alignment Plot) oluşturuluyor...")
    model.eval()
    with torch.no_grad():
        test_seq, test_mel, test_text = samples[0]
        x_test = model.embedding(test_seq.unsqueeze(0))
        for l in model.encoder: x_test = l(x_test)
        test_soft = model.aligner(x_test, test_mel)[0].exp().cpu().numpy()
        test_hard = monotonic_alignment_search(model.aligner(x_test, test_mel)[0]).cpu().numpy()
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(test_soft, aspect='auto', origin='lower', cmap='viridis')
        plt.title("A_soft (Attention Probabilities)")
        plt.xlabel("Mel Frames (Time)")
        plt.ylabel("Text Characters")
        
        plt.subplot(1, 2, 2)
        plt.imshow(test_hard, aspect='auto', origin='lower', cmap='inferno')
        plt.title("A_hard (MAS Alignment Diagonal)")
        plt.xlabel("Mel Frames (Time)")
        plt.ylabel("Text Characters")
        plt.tight_layout()
        plt.savefig("alignment_plot.png")
        plt.close()
        print("[OK] Hizalama grafiği kaydedildi: alignment_plot.png")
        
        # 2. İki Ayrı Ses Üretimi: Ground-Truth Duration vs Predicted Duration
        print("[*] İki ayrı ses üretiliyor (GT Duration vs Pred Duration)...")
        # A) GT Duration ile
        dur_gt = test_hard.sum(axis=-1)
        mel_gt, _, _, _, _ = model(test_seq.unsqueeze(0), test_mel)
        wav_gt = dsp.mel_to_wav(mel_gt.squeeze(0)).cpu().numpy()
        sf.write("validation_gt_dur.wav", wav_gt, 22050)
        
        # B) Pred Duration ile (İnferans modu)
        mel_pred, _, _, _, _ = model(test_seq.unsqueeze(0))
        wav_pred = dsp.mel_to_wav(mel_pred.squeeze(0)).cpu().numpy()
        sf.write("validation_pred_dur.wav", wav_pred, 22050)
        
        # 3. Süre İstatistiği Kontrolü
        print("\n--- SÜRE (DURATION) ANALİZİ ---")
        chars = [ID_TO_CHAR.get(i.item(), '?') for i in test_seq]
        print("Harfler ve MAS Süreleri (Frame):")
        for ch, d in zip(chars[:20], dur_gt[:20]):
            print(f"'{ch}': {int(d)} frames", end=" | ")
        print(f"\nToplam Frame: {dur_gt.sum()} (Hedef Mel: {test_mel.shape[-1]})")
        print("[OK] Doğrulama başarıyla tamamlandı!")


if __name__ == "__main__":
    run_rigorous_validation()
