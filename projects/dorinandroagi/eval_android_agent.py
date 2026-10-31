import os
import sys
import torch
from tokenizers import Tokenizer
from kaggle_v4_shard5_kernel.main import DorinaDenseV4, DenseConfig

# Windows UTF-8 Terminal Destegi
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def test_android_model():
    print("=" * 75)
    print("📱 DORINA-V4 ANDROID OS AGENT CANLI TEST VE DEGERLENDIRME")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_path = "tokenizers/dorina_tokenizer_v4_sft.json"
    tok = Tokenizer.from_file(tok_path)

    weights_path = "models/dorina_v4/dorina_v4_android_agent.pt"
    if not os.path.exists(weights_path):
        print(f"[!] Hata: {weights_path} henuz tamamlanmadi.")
        return

    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = sd["model"] if "model" in sd else sd

    cfg = DenseConfig(vocab_size=32832, max_seq_len=2048)
    model = DorinaDenseV4(cfg)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.to(device)
    model.eval()

    test_commands = [
        # 1. Alarm
        "Yarın sabah saat 07:45 için alarm kur",
        # 2. Fener
        "Karanlık oldu feneri açar mısın",
        # 3. Ses
        "Telefonun sesini %50 yap",
        # 4. Parlaklık
        "Ekran parlaklığını %20 yap",
        # 5. Batarya
        "Şarjım kaç kaldı?",
        # 6. Uygulama
        "Spotify uygulamasını aç",
        # 7. WhatsApp Mesaj
        "Annem kişisine WhatsApp'tan 'Geliyorum' yaz",
        # 8. Müzik
        "Müziği durdur",
        # 9. Wi-Fi
        "Wi-Fi'ı kapat",
        # 10. Arama
        "Babam kişisini ara",
        # 11. Normal Sohbet (Negatif Test)
        "Merhaba nasılsın?",
        # 12. Kimlik
        "Sen kimsin?"
    ]

    im_end_id = tok.token_to_id("<|im_end|>")
    tool_end_id = tok.token_to_id("<|tool_call_end|>")

    for cmd in test_commands:
        prompt = f"<|im_start|>user\n{cmd}<|im_end|>\n<|im_start|>assistant\n"
        enc = tok.encode(prompt)
        inp = torch.tensor([enc.ids], dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(80):
                logits, _ = model(inp)
                nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                inp = torch.cat([inp, nxt], dim=1)
                tok_id = nxt.item()
                if tok_id in [im_end_id, tool_end_id, 0]:
                    break

        full = tok.decode(inp[0].tolist())
        ans = full.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
        print(f"\n[TALHA]: {cmd}")
        print(f"[DORINA]: {ans}")

if __name__ == "__main__":
    test_android_model()
