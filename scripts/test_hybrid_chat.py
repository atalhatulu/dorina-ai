import sys
import torch
from tokenizers import Tokenizer
sys.path.insert(0, 'kaggle_kernel')
from model_moe_v3 import DorinaMoEV3, MoEConfig

tok = Tokenizer.from_file('kaggle_upload/dorina-v3-dataset/dorina_tokenizer_v3_32k.json')
cfg = MoEConfig()
model = DorinaMoEV3(cfg).cuda()
sd = torch.load('models/dorina_v3_titan_hybrid_4500_7500.pt', map_location='cuda', weights_only=False)
model.load_state_dict(sd)
model.eval()

prompts = [
    '<|im_start|>user\nMerhaba, sen kimsin ve ne yapabilirsin?<|im_end|>\n<|im_start|>assistant\n',
    'def reverse_list(items):\n    """Reverse the list in Python."""\n',
    'Question: Python bir listenin elemanlarini tersine nasil ceviririz?\nAnswer:',
    'Turkiye baskenti neresidir?'
]

for p in prompts:
    print('=' * 60)
    print('PROMPT:', p.strip())
    ids = torch.tensor([tok.encode(p).ids], device='cuda')
    out = model.generate(ids, max_new_tokens=60, temperature=0.7, top_k=40)
    gen_text = tok.decode(out[0].tolist())
    resp = gen_text[len(p):]
    print('MELEZ MODEL CIKTISI:\n' + resp.strip())
    print('=' * 60)
