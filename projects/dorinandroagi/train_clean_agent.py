import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from kaggle_v4_shard5_kernel.main import DorinaDenseV4, DenseConfig

# Windows UTF-8 Terminal Fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

class CleanAgentDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len=256):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.samples = []
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"

        for item in raw_data:
            msgs = item["messages"]
            user_msg = msgs[0]["content"]
            asst_msg = msgs[1]["content"]

            prompt_text = f"{im_start}user\n{user_msg}{im_end}\n{im_start}assistant\n"
            target_text = f"{asst_msg}{im_end}"

            prompt_enc = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            target_enc = tokenizer(target_text, add_special_tokens=False)["input_ids"]

            input_ids = (prompt_enc + target_enc)[:max_seq_len]
            # Label masking: mask user prompt with -100
            labels = ([-100] * len(prompt_enc) + target_enc)[:max_seq_len]

            # Padding
            pad_len = max_seq_len - len(input_ids)
            input_ids += [0] * pad_len
            labels += [-100] * pad_len

            self.samples.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train_clean_agent():
    print("=" * 65)
    print(" DORINA-V4 NET & DOĞRUDAN ASİSTAN SFT EĞİTİMİ (RTX 4050)")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = PreTrainedTokenizerFast(tokenizer_file="tokenizers/dorina_tokenizer_v4_sft.json")
    if tok.pad_token_id is None:
        tok.pad_token_id = 0

    dataset = CleanAgentDataset("data/dorina_gold_sft_v4/dorina_clean_direct_agent.json", tok, max_seq_len=128)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    print(f"[+] Veri Seti: {len(dataset)} adet kristal net komut & diyalog.")

    # Load Base resized 32832 model
    cfg = DenseConfig(vocab_size=32832, max_seq_len=1024)
    model = DorinaDenseV4(cfg)
    base_ckpt = "models/dorina_v4/dorina_v4_shard6_sft_base_resized.pt"
    sd = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    epochs = 2
    total_steps = len(loader) * epochs
    step = 0

    print(f"[*] Egitim basliyor... Toplam adim: {total_steps}")
    for epoch in range(epochs):
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()

            out = model(input_ids)
            logits = out[0] if isinstance(out, tuple) else out

            # Causal shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = loss_fn(shift_logits.view(-1, 32832), shift_labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            if step % 100 == 0 or step == total_steps:
                print(f"[Epoch {epoch+1}/{epochs}] Step {step}/{total_steps} | Loss: {loss.item():.4f}")

    # Save trained model
    out_pt = "models/dorina_v4/dorina_v4_clean_agent.pt"
    torch.save({"model": model.state_dict(), "vocab_size": 32832}, out_pt)
    print(f"\n[+] Egitim Kusursuz Tamamlandi! Kaydedildi: {out_pt}")

if __name__ == "__main__":
    train_clean_agent()
