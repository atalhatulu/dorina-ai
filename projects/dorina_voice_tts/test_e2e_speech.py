import os, sys, torch
import soundfile as sf
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))

from voice.hifigan_model import HiFiGANGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Cihaz: {device}")

# Kaggle notebook'undaki mimarinin birebir kopyasi
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SIL>"]
TURKISH_CHARS = list("abcçdefgğhıijklmnoöprsştuüvyz")
EXTENDED_CHARS = list("âîû")
PUNCTUATIONS = list(".,!? -:;")
VOCAB = SPECIAL_TOKENS + TURKISH_CHARS + EXTENDED_CHARS + PUNCTUATIONS
CHAR_TO_ID = {ch: idx for idx, ch in enumerate(VOCAB)}

def text_to_sequence(text: str) -> list[int]:
    text = text.replace("İ", "i").replace("I", "ı").lower().strip()
    seq = [CHAR_TO_ID["<BOS>"]]
    for ch in text:
        if ch in CHAR_TO_ID: seq.append(CHAR_TO_ID[ch])
        elif ch == " ": seq.append(CHAR_TO_ID.get(" ", CHAR_TO_ID["<SIL>"]))
        else: seq.append(CHAR_TO_ID["<UNK>"])
    seq.append(CHAR_TO_ID["<EOS>"])
    return seq

class FFTBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, conv_filter_size=512, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, conv_filter_size, 9, padding=4)
        self.conv2 = nn.Conv1d(conv_filter_size, d_model, 1)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        res = x; a, _ = self.attn(x, x, x); x = self.norm1(res + self.drop(a))
        res = x; out = F.relu(self.conv1(x.transpose(1, 2))); out = self.conv2(out).transpose(1, 2)
        return self.norm2(res + self.drop(out))

class VariancePredictor(nn.Module):
    def __init__(self, d_model=256, filter_size=256):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, filter_size, 3, padding=1)
        self.norm1 = nn.LayerNorm(filter_size)
        self.conv2 = nn.Conv1d(filter_size, filter_size, 3, padding=1)
        self.norm2 = nn.LayerNorm(filter_size)
        self.linear = nn.Linear(filter_size, 1)
    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x.transpose(1, 2)).transpose(1, 2)))
        out = F.relu(self.norm2(self.conv2(out.transpose(1, 2)).transpose(1, 2)))
        return self.linear(out).squeeze(-1)

class LengthRegulator(nn.Module):
    def forward(self, x, durs):
        output = []
        for b in range(x.size(0)):
            seq, d = x[b], durs[b]
            exp = [token.repeat(max(1, int(round(dur.item()))), 1) for token, dur in zip(seq, d)]
            output.append(torch.cat(exp, dim=0))
        max_l = max(o.size(0) for o in output)
        padded = [F.pad(o, (0, 0, 0, max_l - o.size(0))) if max_l > o.size(0) else o for o in output]
        return torch.stack(padded)

class KaggleFastPitch(nn.Module):
    def __init__(self, vocab_size=len(VOCAB), d_model=256, layers=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.encoder = nn.ModuleList([FFTBlock(d_model) for _ in range(layers)])
        self.mel_conv = nn.Sequential(nn.Conv1d(80, d_model, 3, padding=1), nn.ReLU(), nn.Conv1d(d_model, d_model, 1))
        self.dur_pred = VariancePredictor(d_model)
        self.pitch_pred = VariancePredictor(d_model)
        self.pitch_emb = nn.Conv1d(1, d_model, 9, padding=4)
        self.lr = LengthRegulator()
        self.decoder = nn.ModuleList([FFTBlock(d_model) for _ in range(layers)])
        self.mel_linear = nn.Linear(d_model, 80)

    def forward(self, seq):
        x = self.emb(seq)
        for l in self.encoder: x = l(x)
        d_pred = self.dur_pred(x)
        p_pred = self.pitch_pred(x)
        d_use = torch.clamp(torch.round(torch.exp(d_pred) - 1.0), min=1, max=25)
        x = x + self.pitch_emb(p_pred.unsqueeze(1)).transpose(1, 2)
        x = self.lr(x, d_use)
        for l in self.decoder: x = l(x)
        mel_out = self.mel_linear(x).transpose(1, 2)
        return mel_out, d_pred, d_use, p_pred

# Modelleri Yükle
fastpitch = KaggleFastPitch().to(device)
ckpt_path = "models/fastpitch_checkpoint/checkpoints/dorina_fastpitch_step_1000.pt"
print(f"[*] Kaggle FastPitch yukleniyor: {ckpt_path}")
fp_weights = torch.load(ckpt_path, map_location=device)
fastpitch.load_state_dict(fp_weights)
fastpitch.eval()

vocoder = HiFiGANGenerator().to(device)
vocoder_ckpt_path = "models/kaggle_checkpoint/checkpoints/dorina_hifigan_active.pt"
print(f"[*] HiFi-GAN yukleniyor: {vocoder_ckpt_path}")
voc_data = torch.load(vocoder_ckpt_path, map_location=device)
if "generator" in voc_data:
    vocoder.load_state_dict(voc_data["generator"])
else:
    vocoder.load_state_dict(voc_data)
vocoder.eval()

def generate_voice(text, filename):
    print(f"\n[+] Sentezleniyor: '{text}'")
    with torch.no_grad():
        seq = torch.tensor(text_to_sequence(text), dtype=torch.long).unsqueeze(0).to(device)
        mel_out, d_pred, d_use, p_pred = fastpitch(seq)
        print(f"    Mel Sekli: {mel_out.shape} | Tahmini Sure: {mel_out.shape[-1] * 256 / 22050:.2f} sn")
        print(f"    Mel Min: {mel_out.min().item():.2f}, Max: {mel_out.max().item():.2f}, Mean: {mel_out.mean().item():.2f}")
        
        audio = vocoder(mel_out).squeeze().cpu().numpy()
        # Max normalizasyon
        audio = audio / max(np.max(np.abs(audio)), 1e-6) * 0.95
        sf.write(filename, audio, 22050)
        print(f"    [OK] Basariyla olusturuldu: {filename}")

generate_voice("merhaba talha ben dorina", "dorina_speech_1000_talha.wav")
generate_voice("kendi sesimle konusuyorum", "dorina_speech_1000_konusma.wav")
print("\n[TEBRIKLER] Uctan uca ilk modellerle ses dosyasi uretildi!")
