import os
import sys
import time
import subprocess
import torch
from dataclasses import dataclass

sys.stdout.reconfigure(encoding='utf-8')

env = os.environ.copy()
env['KAGGLE_API_TOKEN'] = 'KGAT_4d3d2d7d549486b6a902e30127c0184f'
env['KAGGLE_CONFIG_DIR'] = 'C:/Users/Talha/temp_kaggle'

KERNEL_SLUG = 'abdullahtalhatl/dorina-nano-tr-500m-dual-t4-ddp'
DOWNLOAD_DIR = 'kaggle_nano_500m_download'
REPORT_FILE = 'dorina_nano_500m_final_report.txt'

print("=" * 65)
print("DORINA-NANO-TR (10M - 500M TOKEN) FINISHER & AUTO-SHUTDOWN WATCHDOG")
print("Target: Wait for COMPLETE -> Download -> Test -> Log -> Safe Shutdown")
print("=" * 65, flush=True)

while True:
    res = subprocess.run(['kaggle', 'kernels', 'status', KERNEL_SLUG], env=env, capture_output=True, text=True)
    status_str = res.stdout.strip()
    now_str = time.strftime('%H:%M:%S')
    print(f"[{now_str}] Status: {status_str}", flush=True)
    
    if "COMPLETE" in status_str or "CANCEL_ACKNOWLEDGED" in status_str:
        print("\n>>> KERNEL FINISHED! Starting Download...", flush=True)
        break
    elif "ERROR" in status_str:
        print("\n>>> KERNEL REPORTED ERROR! Inspecting...", flush=True)
        break
        
    time.sleep(60)

# Download output
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
subprocess.run(['kaggle', 'kernels', 'output', KERNEL_SLUG, '-p', DOWNLOAD_DIR], env=env)
print(">>> Output downloaded successfully!", flush=True)

# Run Comprehensive Evaluation
from scripts.dorina_nano_tr_model import DorinaNanoTR, NanoTRConfig
from tokenizers import Tokenizer

tok_path = "tokenizers/dorina_nano_tr_12k.json"
tokenizer = Tokenizer.from_file(tok_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = NanoTRConfig(
    vocab_size=12288,
    max_seq_len=256,
    d_model=256,
    n_layer=10,
    n_heads=8,
    n_kv_heads=4,
    head_dim=32,
    intermediate_size=688,
    tied_embeddings=True
)

# Find latest checkpoint
target_pt = None
for root, dirs, files in os.walk(DOWNLOAD_DIR):
    for f in files:
        if f.endswith('.pt'):
            target_pt = os.path.join(root, f)

report_lines = []
report_lines.append("=== DORINA-NANO-TR 10M (500M TOKEN RUN) FINAL TEST REPORT ===")
report_lines.append(f"Model file: {target_pt}")

if target_pt and os.path.exists(target_pt):
    state_dict = torch.load(target_pt, map_location=device, weights_only=True)
    model = DorinaNanoTR(cfg).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    def generate(prompt, max_new_tokens=30, temp=0.4, top_k=30):
        input_ids = tokenizer.encode(prompt).ids
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            x_cond = x if x.size(1) <= cfg.max_seq_len else x[:, -cfg.max_seq_len:]
            with torch.no_grad():
                res = model(x_cond)
                logits = res[0] if isinstance(res, tuple) else res
                logits = logits[:, -1, :] / temp
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                x = torch.cat((x, idx_next), dim=1)
                if idx_next.item() == tokenizer.token_to_id('<eos>'):
                    break
        return tokenizer.decode(x[0].tolist())

    tests = [
        '<complete> Türkiye Cumhuriyeti',
        '<complete> Yapay zeka teknolojileri',
        '<complete> Bugün hava çok',
        '<fix> bugun hava cok guzel =>',
        '<fix> turkiyenin baskenti ankaradir =>',
        '<fix> herkez ev de kaldi =>',
        '<fix> geliyorum ama biraz gec kalicam =>',
        '<fix> yarin kesinlikle gelcem bişey olmaz =>'
    ]

    for t in tests:
        out = generate(t, max_new_tokens=25, temp=0.35)
        report_lines.append(f"\nPROMPT: {t}")
        report_lines.append(f"OUTPUT: {out}")
        report_lines.append("-" * 50)
else:
    report_lines.append("No .pt model file found in output.")

report_text = "\n".join(report_lines)
with open(REPORT_FILE, "w", encoding="utf-8") as rf:
    rf.write(report_text)

print(report_text, flush=True)
print("\n>>> All tasks finished. Auto-shutdown sequence initiating in 60 seconds...", flush=True)

# Safe Windows shutdown (allows 60 seconds cancellation if needed via 'shutdown /a')
subprocess.run(['shutdown', '/s', '/t', '60'])
