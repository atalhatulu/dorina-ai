import json

notebook_path = "voice/kaggle_fastpitch/dorina-fastpitch-train.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell4_code = """# 3. FastPitch Egitim Dongusu (10.000 Adim, GPU + FP16 + OOM Onleyici)
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

# 1.0 sn - 10.0 sn arasi guvenli sesleri filtrele (OOM onlemi)
lines = []
for fn, txt in raw_lines:
    wav_path = f"dataset/wavs/{fn}"
    if os.path.exists(wav_path):
        try:
            info = sf.info(wav_path)
            duration = info.frames / info.samplerate
            if 1.0 <= duration <= 10.0 and len(txt.strip()) > 3:
                lines.append((fn, txt))
        except Exception:
            pass

print(f"[*] Filtrelenmis temiz veri sayisi: {len(lines)}")

TOTAL_STEPS = 10000
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
    
    if step % 100 == 0 or step == 1:
        print(f"Step [{step}/{TOTAL_STEPS}] - Mel Loss: {loss_mel.item():.4f} | Dur Loss: {loss_dur.item():.4f}", flush=True)
        
    if step % 2500 == 0 or step == TOTAL_STEPS:
        torch.save(model.state_dict(), f"checkpoints/dorina_fastpitch_step_{step}.pt")
        print(f"[CHECKPOINT] Adim {step} kaydedildi.", flush=True)

print("[OK] FastPitch egitimi basariyla tamamlandi!")
"""

nb["cells"][4]["source"] = [line + "\n" for line in cell4_code.split("\n")]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Updated notebook successfully!")
