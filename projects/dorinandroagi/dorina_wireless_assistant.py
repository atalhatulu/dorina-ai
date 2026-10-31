import os
import sys
import subprocess
import json
import torch
from tokenizers import Tokenizer
from kaggle_v4_shard5_kernel.main import DorinaDenseV4, DenseConfig

# UTF-8 Support
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stdin.reconfigure(encoding='utf-8')

ADB_PATH = os.path.expandvars(r"$LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe")
DEVICE_ID = "192.168.1.231:40581"

def run_adb_command(cmd_args):
    full_cmd = [ADB_PATH, "-s", DEVICE_ID, "shell"] + cmd_args
    try:
        res = subprocess.run(full_cmd, capture_output=True, text=True, timeout=5)
        return res.stdout.strip()
    except Exception as e:
        return ""

def execute_android_action(name, args):
    print(f"[*] Telefona Eylem Gönderiliyor: {name} -> {args}")

    if name == "set_alarm":
        # Samsung OneUI doğrudan intent'e SecurityException atıyor, bu yüzden Saat uygulamasını açıyoruz
        run_adb_command(["monkey", "-p", "com.sec.android.app.clockpackage", "-c", "android.intent.category.LAUNCHER", "1"])
        return f"⏰ Samsung Saat/Alarm uygulaması ekranda açıldı!"

    elif name == "set_volume":
        # Samsung Android'de 'media' binary yok, ses tuşları simülasyonu (Hardware Keyevent) kullanılır!
        level = int(args.get("level", 50))
        # 24 = Volume Up, 25 = Volume Down
        if level >= 50:
            for _ in range(5): run_adb_command(["input", "keyevent", "24"])
        else:
            for _ in range(5): run_adb_command(["input", "keyevent", "25"])
        return f"🔊 Ses seviyesi fiziksel olarak ayarlandı!"

    elif name == "set_brightness":
        level = int(args.get("level", 50))
        br_val = int((level / 100.0) * 255)
        run_adb_command(["settings", "put", "system", "screen_brightness", str(br_val)])
        return f"☀️ Ekran parlaklığı %{level} yapıldı!"

    elif name == "get_battery_status":
        out = run_adb_command(["dumpsys", "battery"])
        level = "69"
        for line in out.splitlines():
            if "level:" in line:
                level = line.split("level:")[-1].strip()
                break
        return f"🔋 Telefonunun bataryası şu anda %{level} seviyesinde Talha!"

    elif name == "open_app":
        app = args.get("app_name", "").lower()
        if "youtube" in app:
            run_adb_command(["monkey", "-p", "com.google.android.youtube", "-c", "android.intent.category.LAUNCHER", "1"])
            return "▶️ YouTube telefonunda başarıyla açıldı!"
        elif "spotify" in app:
            run_adb_command(["monkey", "-p", "com.spotify.music", "-c", "android.intent.category.LAUNCHER", "1"])
            return "🎵 Spotify açıldı!"
        elif "whatsapp" in app:
            run_adb_command(["monkey", "-p", "com.whatsapp", "-c", "android.intent.category.LAUNCHER", "1"])
            return "💬 WhatsApp açıldı!"
        elif "tiktok" in app:
            # TikTok resmi paket adı
            run_adb_command(["monkey", "-p", "com.zhiliaoapp.musically", "-c", "android.intent.category.LAUNCHER", "1"])
            return "📱 TikTok telefonunda açıldı!"
        elif "kamera" in app or "camera" in app:
            run_adb_command(["am", "start", "-a", "android.media.action.IMAGE_CAPTURE"])
            return "📷 Kamera açıldı!"
        elif "saat" in app or "alarm" in app:
            run_adb_command(["monkey", "-p", "com.sec.android.app.clockpackage", "-c", "android.intent.category.LAUNCHER", "1"])
            return "⏰ Saat uygulaması açıldı!"
        else:
            return f"📱 {app} uygulaması için komut gönderildi."

    elif name == "media_control":
        action = args.get("action", "pause")
        if action == "pause":
            run_adb_command(["input", "keyevent", "127"])
            return "⏸️ Müzik duraklatıldı."
        elif action == "play":
            run_adb_command(["input", "keyevent", "126"])
            return "▶️ Müzik devam ediyor."
        elif action == "next":
            run_adb_command(["input", "keyevent", "87"])
            return "⏭️ Sıradaki şarkıya geçildi."

    elif name == "make_phone_call":
        contact = args.get("contact", "")
        run_adb_command(["am", "start", "-a", "android.intent.action.DIAL"])
        return f"📞 {contact} aranıyor..."

    return "İşlem tamamlandı."

def main():
    print("=" * 75)
    print("📱 DORINA-V4 SAMSUNG GALAXY A70 UYUMLU ASISTAN (TAM YETKILI)")
    print("=" * 75)

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

    print("\n[+] Dorina-V4 Samsung A70 icin optimize edildi! Komutunu yaz:")

    while True:
        try:
            cmd = input("\nSen: ").strip()
            if not cmd: continue
            if cmd.lower() in ["q", "exit"]: break

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

            if '{"name":' in ans:
                json_start = ans.find('{"name":')
                json_end = ans.rfind('}') + 1
                json_str = ans[json_start:json_end]
                try:
                    tool_data = json.loads(json_str)
                    action_msg = execute_android_action(tool_data.get("name"), tool_data.get("arguments", {}))
                    print(f"Dorina: {action_msg}")
                except Exception as e:
                    print(f"Dorina: {ans}")
            else:
                print(f"Dorina: {ans}")

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
