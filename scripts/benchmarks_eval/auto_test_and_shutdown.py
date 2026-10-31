import os
import sys
import time
import subprocess
import torch
from scripts.dorina_nano_tr_model import DorinaNanoTR, NanoTRConfig
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding='utf-8')

target_model = 'models/dorina_nano_tr/dorina_nano_tr_10m_500m.pt'
while not os.path.exists(target_model):
    time.sleep(3)

time.sleep(3)

print('>>> 500M MODEL DETECTED! Running comprehensive test...', flush=True)

tokenizer = Tokenizer.from_file('tokenizers/dorina_nano_tr_12k.json')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = NanoTRConfig(
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

state_dict = torch.load(target_model, map_location=device, weights_only=True)
model = DorinaNanoTR(config).to(device)
model.load_state_dict(state_dict)
model.eval()

def generate(prompt, max_new_tokens=30, temp=0.35, top_k=30):
    input_ids = tokenizer.encode(prompt).ids
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        x_cond = x if x.size(1) <= config.max_seq_len else x[:, -config.max_seq_len:]
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

report_lines = []
report_lines.append('=== DORINA-NANO-TR 10M (500M GRAND FINALE TEST RAPORU) ===')
report_lines.append('=' * 65)

for t in tests:
    out = generate(t, max_new_tokens=25, temp=0.35)
    report_lines.append(f'PROMPT: {t}')
    report_lines.append(f'OUTPUT: {out}')
    report_lines.append('-' * 65)

report_text = '\n'.join(report_lines)
with open('dorina_nano_500m_final_report.txt', 'w', encoding='utf-8') as rf:
    rf.write(report_text)

print(report_text, flush=True)
print('\n>>> All tests completed! Shutting down in 30 seconds...', flush=True)
subprocess.run(['shutdown', '/s', '/t', '30'])
