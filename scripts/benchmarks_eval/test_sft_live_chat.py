import sys
import torch
from tokenizers import Tokenizer
sys.path.insert(0, 'kaggle_kernel')
from model_moe_v3 import DorinaMoEV3, MoEConfig

tok = Tokenizer.from_file('kaggle_upload/dorina-v3-dataset/dorina_tokenizer_v3_32k.json')
cfg = MoEConfig()
model = DorinaMoEV3(cfg).cuda()
sd = torch.load('models/sft_v3/dorina_v3_sft_epoch_3.pt', map_location='cuda', weights_only=False)
model.load_state_dict(sd)
model.eval()

prompts = [
    '<|im_start|>user\nMerhaba, sen kimsin ve ne yapabilirsin?<|im_end|>\n<|im_start|>assistant\n',
    '<|im_start|>user\nPython ile bir listenin elemanlarini tersine ceviren bir fonksiyon yazar misin?<|im_end|>\n<|im_start|>assistant\n',
    '<|im_start|>user\nTurkiye\'nin baskenti neresidir?<|im_end|>\n<|im_start|>assistant\n'
]

eos_id = tok.token_to_id('<|im_end|>')

for p in prompts:
    print('=' * 60)
    print('PROMPT:', p.strip())
    ids = torch.tensor([tok.encode(p).ids], device='cuda')
    out = model.generate(ids, max_new_tokens=80, temperature=0.6, top_k=30, eos_id=eos_id)
    gen_text = tok.decode(out[0].tolist())
    resp = gen_text[len(p):]
    print('DORINA V3 SFT YANITI:\n' + resp.strip())
    print('=' * 60)
