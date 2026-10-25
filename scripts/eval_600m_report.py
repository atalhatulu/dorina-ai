import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from scripts.dorina_nano_tr_model import DorinaNanoTR, NanoTRConfig
from tokenizers import Tokenizer
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')

tok_path = 'tokenizers/dorina_nano_tr_12k.json'
model_path = 'models/dorina_nano_tr/dorina_nano_tr_10m_600m.pt'

tokenizer = Tokenizer.from_file(tok_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

state_dict = torch.load(model_path, map_location=device, weights_only=True)
model = DorinaNanoTR(cfg).to(device)
model.load_state_dict(state_dict)
model.eval()

def predict_next_tokens(text_prompt, top_k=5):
    full_prompt = f'<complete> {text_prompt}'
    input_ids = tokenizer.encode(full_prompt).ids
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        res = model(x)
        logits = res[0] if isinstance(res, tuple) else res
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
    return [(tokenizer.decode([idx.item()]), prob.item() * 100) for prob, idx in zip(top_probs, top_indices)]

def fix_sentence(corrupted_text, max_tokens=30):
    fix_prompt = f'<fix> {corrupted_text} =>'
    input_ids = tokenizer.encode(fix_prompt).ids
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    gen = []
    for _ in range(max_tokens):
        x_cond = x if x.size(1) <= cfg.max_seq_len else x[:, -cfg.max_seq_len:]
        with torch.no_grad():
            res = model(x_cond)
            logits = res[0] if isinstance(res, tuple) else res
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            x = torch.cat((x, next_id), dim=1)
            tok_id = next_id.item()
            if tok_id == tokenizer.token_to_id('<eos>') or tok_id == 3:
                break
            tok_char = tokenizer.decode([tok_id])
            if '\n' in tok_char:
                break
            gen.append(tok_id)
    return tokenizer.decode(gen).strip()

def complete_sentence(text_prompt, max_tokens=25, temp=0.35):
    full_prompt = f'<complete> {text_prompt}'
    input_ids = tokenizer.encode(full_prompt).ids
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    gen = []
    for _ in range(max_tokens):
        x_cond = x if x.size(1) <= cfg.max_seq_len else x[:, -cfg.max_seq_len:]
        with torch.no_grad():
            res = model(x_cond)
            logits = (res[0] if isinstance(res, tuple) else res)[:, -1, :] / temp
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_id), dim=1)
            tok_id = next_id.item()
            if tok_id == tokenizer.token_to_id('<eos>') or tok_id == 3:
                break
            gen.append(tok_id)
    return tokenizer.decode(gen).strip()

print('=== DORINA-NANO-TR 10M (600M KİLOMETRE TAŞI DERİN RAPORU) ===')
print('=' * 65)

print('\n>>> [1] SONRAKİ KELİME TAHMİNLERİ (% OLASILIKLAR):')
tests_pct = ['Bugün hava çok', 'Türkiye Cumhuriyeti', 'Yapay zeka']
for p in tests_pct:
    cands = predict_next_tokens(p, top_k=4)
    print(f'Prompt: "{p}"')
    for r, (tok, pct) in enumerate(cands, 1):
        bar = '█' * int(pct / 2)
        print(f'   {r}. {repr(tok):<14} %{pct:5.2f}  {bar}')

print('\n>>> [2] CÜMLE VE İMLA DÜZELTME TESTLERİ (<fix>):')
fix_tests = [
    'bugun hava cok guzel',
    'turkiyenin baskenti ankaradir',
    'herkez ev de kaldi',
    'geliyorum ama biraz gec kalicam',
    'yarin kesinlikle gelcem bişey olmaz'
]
for f in fix_tests:
    print(f'{f:<35} => {fix_sentence(f)}')

print('\n>>> [3] CÜMLE TAMAMLAMA TESTLERİ (<complete>):')
comp_tests = ['Türkiye Cumhuriyeti', 'Yapay zeka modelleri', 'Yarın sabah erken']
for c in comp_tests:
    print(f'"{c}" => {complete_sentence(c)}')
