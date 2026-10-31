import os
import sys
import torch
from tokenizers import Tokenizer
from kaggle_v4_shard5_kernel.main import DorinaDenseV4, DenseConfig

# Windows UTF-8 Terminal Destegi
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stdin.reconfigure(encoding='utf-8')

def main():
    print("=" * 70)
    print("CANLI ETKILESIMLI ANDROID TESTI (SIFIR HILE / GERCEK INFERENCE)")
    print("Cikis icin 'q' yaz.")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_path = "tokenizers/dorina_tokenizer_v4_sft.json"
    tok = Tokenizer.from_file(tok_path)

    weights_path = "models/dorina_v4/dorina_v4_android_agent.pt"
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = sd["model"] if "model" in sd else sd

    cfg = DenseConfig(vocab_size=32832, max_seq_len=2048)
    model = DorinaDenseV4(cfg)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.to(device)
    model.eval()

    im_end_id = tok.token_to_id("<|im_end|>")
    tool_end_id = tok.token_to_id("<|tool_call_end|>")

    print("[+] Model GPU'da hazir bekliyor. Klavyeden istedigin komutu yaz!\n")

    while True:
        try:
            cmd = input("\nSen: ").strip()
            if not cmd: continue
            if cmd.lower() in ["q", "exit"]: break

            prompt = f"<|im_start|>user\n{cmd}<|im_end|>\n<|im_start|>assistant\n"
            enc = tok.encode(prompt)
            inp = torch.tensor([enc.ids], dtype=torch.long, device=device)

            print("Dorina-Android: ", end="", flush=True)

            with torch.no_grad():
                for _ in range(80):
                    logits, _ = model(inp)
                    # Greedy search - Modelin kendi saf tahmini
                    nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    inp = torch.cat([inp, nxt], dim=1)
                    tok_id = nxt.item()
                    if tok_id in [im_end_id, tool_end_id, 0]:
                        break
                    
                    t_str = tok.decode([tok_id])
                    print(t_str, end="", flush=True)
            print()

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
