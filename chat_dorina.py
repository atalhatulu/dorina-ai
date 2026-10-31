import os
import sys
import torch
from tokenizers import Tokenizer
from kaggle_v4_shard5_kernel.main import DorinaDenseV4, DenseConfig

def run_chat():
    print("=" * 75)
    print("🌟 DORINA-V4 CANLI INTERAKTIF SOHBET (144.7M SFT MODELI)")
    print("   Cikis icin 'q' veya 'exit' yazabilirsin.")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_path = "tokenizers/dorina_tokenizer_v4_sft.json"
    tok = Tokenizer.from_file(tok_path)

    weights_path = "models/dorina_v4/dorina_v4_gold_sft_final.pt"
    if not os.path.exists(weights_path):
        print(f"[!] Hata: {weights_path} bulunamadi!")
        return

    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = sd["model"] if "model" in sd else sd

    cfg = DenseConfig(vocab_size=32832, max_seq_len=2048)
    model = DorinaDenseV4(cfg)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.to(device)
    model.eval()

    im_end_id = tok.token_to_id("<|im_end|>")
    tool_end_id = tok.token_to_id("<|tool_call_end|>")

    history = ""
    print(f"\n[+] Dorina-V4 hazir! ({device.upper()} uzerinde calisiyor)\n")

    while True:
        try:
            user_input = input("\nSen: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["q", "exit", "quit"]:
                print("\nDorina: Gorusmek uzere Talha! Kendine iyi bak. 👋\n")
                break

            prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            enc = tok.encode(prompt)
            inp = torch.tensor([enc.ids], dtype=torch.long, device=device)

            print("Dorina: ", end="", flush=True)

            generated_tokens = []
            with torch.no_grad():
                for _ in range(150):
                    logits, _ = model(inp)
                    # Sampling with soft temp
                    next_token_logits = logits[0, -1, :] / 0.7
                    
                    # Top-k filtreleme (k=40)
                    top_k_val = min(40, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_val)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float("Inf")
                    
                    probs = torch.softmax(next_token_logits, dim=-1)
                    nxt = torch.multinomial(probs, num_samples=1).unsqueeze(0)
                    
                    inp = torch.cat([inp, nxt], dim=1)
                    tok_id = nxt.item()

                    if tok_id in [im_end_id, tool_end_id, 0]:
                        break

                    token_str = tok.decode([tok_id])
                    safe_str = token_str.encode("ascii", "replace").decode("ascii")
                    print(safe_str, end="", flush=True)
                    generated_tokens.append(tok_id)

            print()

        except KeyboardInterrupt:
            print("\nDorina: Sohbet sonlandirildi.")
            break

if __name__ == "__main__":
    run_chat()
