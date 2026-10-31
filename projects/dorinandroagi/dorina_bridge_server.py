import re
import json
from aiohttp import web
import torch
import sys
from tokenizers import Tokenizer
from kaggle_v4_shard5_kernel.main import DorinaDenseV4, DenseConfig

# Windows UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

print("[*] Dorina-V4 Yeni Temiz Asistan Modeli Yukleniyor...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok_path = "tokenizers/dorina_tokenizer_v4_sft.json"
tokenizer = Tokenizer.from_file(tok_path)

cfg = DenseConfig(vocab_size=32832, max_seq_len=512)
model = DorinaDenseV4(cfg)
ckpt_path = "models/dorina_v4/dorina_v4_clean_agent.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"], strict=True)
model.to(device)
model.eval()
print("[+] Dorina-V4 Temiz Model GPU'da Calisiyor!")

def generate_answer(query: str):
    prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    enc = tokenizer.encode(prompt)
    curr = list(enc.ids)

    for _ in range(50):
        inp = torch.tensor([curr[-512:]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(inp)
            logits = out[0] if isinstance(out, tuple) else out
        nt = torch.argmax(logits[:, -1, :], dim=-1).item()
        curr.append(nt)
        if nt in [2, tokenizer.token_to_id("<|im_end|>")]:
            break

    raw_response = tokenizer.decode(curr[len(enc.ids):]).replace("<|im_end|>", "").strip()
    return raw_response

async def handle_inference(request):
    try:
        data = await request.json()
        query = data.get("query", "").strip()
        print(f"\n[A70'ten Gelen Komut]: '{query}'")

        raw = generate_answer(query)
        print(f"[Dorina-V4 Yaniti]: '{raw}'")

        # Check if JSON tool call
        if raw.startswith("{") and "name" in raw:
            try:
                tool_data = json.loads(raw)
                return web.json_response({
                    "type": "tool",
                    "tool": tool_data.get("name"),
                    "args": tool_data.get("arguments", {})
                })
            except:
                pass
        
        # Regex fallback for tool
        m = re.search(r'\{.*\}', raw)
        if m:
            try:
                tool_data = json.loads(m.group(0))
                if "name" in tool_data:
                    return web.json_response({
                        "type": "tool",
                        "tool": tool_data.get("name"),
                        "args": tool_data.get("arguments", {})
                    })
            except:
                pass

        # Otherwise plain dialogue text
        return web.json_response({
            "type": "text",
            "text": raw
        })

    except Exception as e:
        print(f"[-] Hata: {e}")
        return web.json_response({"error": str(e)}, status=500)

app = web.Application()
app.router.add_post('/inference', handle_inference)

if __name__ == "__main__":
    print("[*] Canli Asistan Sunucusu 0.0.0.0:8080 uzerinde baslatildi!")
    web.run_app(app, host='0.0.0.0', port=8080)
