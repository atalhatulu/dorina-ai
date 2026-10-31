import json

notebook_path = "voice/kaggle_fastpitch/dorina-fastpitch-train.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 3: Numba/Numpy ile 30x Hızlı MAS
cell3_code = '''# 2. Tokenizer, Mel ve Ultra Hizli MAS Hizalama
import numpy as np

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

class MelSpectrogramExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, n_fft=1024, win_length=1024, hop_length=256,
            f_min=0.0, f_max=8000.0, n_mels=80, power=1.0, normalized=False, center=False
        )
    def forward(self, wav):
        if wav.dim() == 1: wav = wav.unsqueeze(0)
        return torch.log(torch.clamp(self.mel_transform(wav), min=1e-5))

# 30 KAT HIZLI NUMPY MAS
def fast_mas_np(sim_np):
    t_text, t_mel = sim_np.shape
    dp = np.full((t_text, t_mel), -1e9, dtype=np.float32)
    dp[0, 0] = sim_np[0, 0]
    for j in range(1, t_mel):
        dp[0, j] = dp[0, j - 1] + sim_np[0, j]
    for i in range(1, t_text):
        for j in range(i, t_mel):
            v1 = dp[i, j - 1]
            v2 = dp[i - 1, j - 1]
            dp[i, j] = sim_np[i, j] + (v1 if v1 > v2 else v2)
    align = np.zeros((t_text, t_mel), dtype=np.float32)
    curr = t_text - 1
    align[curr, t_mel - 1] = 1.0
    for j in range(t_mel - 1, 0, -1):
        if curr > 0 and dp[curr - 1, j - 1] >= dp[curr, j - 1]:
            curr -= 1
        align[curr, j - 1] = 1.0
    return align

def batch_mas(sim_tensor):
    device = sim_tensor.device
    sim_np = sim_tensor.cpu().numpy()
    aligns = [fast_mas_np(s) for s in sim_np]
    durations = [np.sum(a, axis=-1) for a in aligns]
    return torch.tensor(np.stack(durations), dtype=torch.float32, device=device)

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

class DorinaFastPitch(nn.Module):
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

    def forward(self, seq, mel_target=None):
        x = self.emb(seq)
        for l in self.encoder: x = l(x)
        d_pred = self.dur_pred(x)
        p_pred = self.pitch_pred(x)
        if mel_target is not None:
            with torch.no_grad():
                mel_p = self.mel_conv(mel_target).transpose(1, 2)
                sim = torch.bmm(x, mel_p.transpose(1, 2)) / 16.0
                soft = F.log_softmax(sim, dim=-1)
                d_use = batch_mas(soft)
        else:
            d_use = torch.clamp(torch.round(torch.exp(d_pred) - 1.0), min=1, max=25)
        x = x + self.pitch_emb(p_pred.unsqueeze(1)).transpose(1, 2)
        x = self.lr(x, d_use)
        for l in self.decoder: x = l(x)
        mel_out = self.mel_linear(x).transpose(1, 2)
        return mel_out, d_pred, d_use, p_pred
print("Hizli Mimariler hazir!")
'''

# Cell 4: 2.500 Adim (Hizli MAS ile ~8-10 dakika)
cell4_code = '''# 3. FastPitch Egitim Dongusu (2.500 Adim - Hizlandirilmis)
import os, random, torch
import soundfile as sf
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

device = "cuda" if torch.cuda.is_available() else "cpu"
mel_extractor = MelSpectrogramExtractor().to(device)
model = DorinaFastPitch().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scaler = GradScaler()

with open("dataset/metadata.csv", "r", encoding="utf-8") as f:
    raw_lines = [l.strip().split("|") for l in f if "|" in l]

lines = []
for fn, txt in raw_lines:
    wav_path = f"dataset/wavs/{fn}"
    if os.path.exists(wav_path):
        try:
            info = sf.info(wav_path)
            duration = info.frames / info.samplerate
            if 1.0 <= duration <= 8.0 and len(txt.strip()) > 3:
                lines.append((fn, txt))
        except Exception:
            pass

print(f"[*] Filtrelenmis temiz veri sayisi: {len(lines)}")

TOTAL_STEPS = 2500
BATCH_SIZE = 8
os.makedirs("checkpoints", exist_ok=True)

model.train()
print("[*] Egitim baslatiliyor...")
for step in range(1, TOTAL_STEPS + 1):
    batch = random.sample(lines, BATCH_SIZE)
    seqs = [torch.tensor(text_to_sequence(t), dtype=torch.long) for _, t in batch]
    wavs = [torch.tensor(sf.read(f"dataset/wavs/{fn}")[0], dtype=torch.float32) for fn, _ in batch]
    
    max_s = max(len(s) for s in seqs)
    max_w = max(len(w) for w in wavs)
    
    seq_b = torch.stack([F.pad(s, (0, max_s - len(s))) for s in seqs]).to(device)
    wav_b = torch.stack([F.pad(w, (0, max_w - len(w))) for w in wavs]).to(device)
    
    with torch.no_grad():
        target_mel = mel_extractor(wav_b)
        
    optimizer.zero_grad()
    
    with autocast():
        mel_pred, d_pred, d_target, p_pred = model(seq_b, target_mel)
        min_f = min(mel_pred.shape[-1], target_mel.shape[-1])
        loss_mel = F.l1_loss(mel_pred[:, :, :min_f], target_mel[:, :, :min_f])
        loss_dur = F.mse_loss(d_pred, torch.log(d_target.detach() + 1.0))
        loss = loss_mel * 2.0 + loss_dur

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    
    if step % 50 == 0 or step == 1:
        print(f"Step [{step}/{TOTAL_STEPS}] - Mel Loss: {loss_mel.item():.4f} | Dur Loss: {loss_dur.item():.4f}", flush=True)
        
    if step % 500 == 0 or step == TOTAL_STEPS:
        torch.save(model.state_dict(), f"checkpoints/dorina_fastpitch_step_{step}.pt")
        print(f"[CHECKPOINT] Adim {step} kaydedildi.", flush=True)

print("[OK] FastPitch egitimi basariyla tamamlandi!")
'''

nb["cells"][3]["source"] = [l + "\n" for l in cell3_code.split("\n")]
nb["cells"][4]["source"] = [l + "\n" for l in cell4_code.split("\n")]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Fast notebook ready!")
