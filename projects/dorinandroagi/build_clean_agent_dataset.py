import json
import random

# Core Tools Definition
tools_catalog = [
    {
        "name": "set_brightness",
        "description": "Ekran parlaklığını ayarlar",
        "parameters": {"level": "int (0-100)"}
    },
    {
        "name": "set_volume",
        "description": "Medya ve ses seviyesini ayarlar",
        "parameters": {"level": "int (0-100)"}
    },
    {
        "name": "set_alarm",
        "description": "Belirtilen saate alarm kurar",
        "parameters": {"time": "str (HH:MM)", "label": "str"}
    },
    {
        "name": "toggle_flashlight",
        "description": "Flaş/feneri açar veya kapatır",
        "parameters": {"enabled": "bool"}
    },
    {
        "name": "get_battery_status",
        "description": "Pil yüzdesini ve şarj durumunu sorgular",
        "parameters": {}
    },
    {
        "name": "open_app",
        "description": "Uygulama başlatır",
        "parameters": {"app_name": "str"}
    }
]

# Clean, direct, realistic commands without literary fluff
clean_templates = [
    # 1. Parlaklık (Net & Pratik)
    ("parlaklığı %{level} yap", "set_brightness", lambda l: {"level": l}),
    ("ekran parlaklığını {level} yap", "set_brightness", lambda l: {"level": l}),
    ("ekranı %{level} yap", "set_brightness", lambda l: {"level": l}),
    ("parlaklık {level}", "set_brightness", lambda l: {"level": l}),
    ("parlaklığı fulle", "set_brightness", lambda _: {"level": 100}),
    ("ekranı en yüksek parlaklığa al", "set_brightness", lambda _: {"level": 100}),
    ("parlaklığı kıs", "set_brightness", lambda _: {"level": 20}),
    ("ekran ışığını kıs", "set_brightness", lambda _: {"level": 20}),
    ("ekran ışığını aç", "set_brightness", lambda _: {"level": 80}),

    # 2. Ses Seviyesi (Net & Pratik)
    ("sesi %{level} yap", "set_volume", lambda l: {"level": l}),
    ("ses seviyesini {level} yap", "set_volume", lambda l: {"level": l}),
    ("sesi kıs", "set_volume", lambda _: {"level": 20}),
    ("sesi fulle", "set_volume", lambda _: {"level": 100}),
    ("telefonu sessize al", "set_volume", lambda _: {"level": 0}),
    ("sesi kapat", "set_volume", lambda _: {"level": 0}),
    ("sesi aç", "set_volume", lambda _: {"level": 70}),
    ("medya sesini %{level} yap", "set_volume", lambda l: {"level": l}),

    # 3. Alarm (Net & Pratik)
    ("saat {time}'e alarm kur", "set_alarm", lambda t: {"time": t, "label": "Alarm"}),
    ("{time} için alarm kur", "set_alarm", lambda t: {"time": t, "label": "Alarm"}),
    ("sabah {time}'e alarm kur", "set_alarm", lambda t: {"time": t, "label": "Sabah"}),
    ("beni saat {time}'de uyandır", "set_alarm", lambda t: {"time": t, "label": "Uyanma"}),
    ("{time}'e alarm ayarla", "set_alarm", lambda t: {"time": t, "label": "Alarm"}),

    # 4. Fener (Net & Pratik)
    ("feneri aç", "toggle_flashlight", lambda _: {"enabled": True}),
    ("flaşı aç", "toggle_flashlight", lambda _: {"enabled": True}),
    ("el fenerini aç", "toggle_flashlight", lambda _: {"enabled": True}),
    ("ışığı aç", "toggle_flashlight", lambda _: {"enabled": True}),
    ("feneri kapat", "toggle_flashlight", lambda _: {"enabled": False}),
    ("flaşı kapat", "toggle_flashlight", lambda _: {"enabled": False}),
    ("el fenerini söndür", "toggle_flashlight", lambda _: {"enabled": False}),

    # 5. Batarya / Şarj (Net & Pratik)
    ("şarjım kaç?", "get_battery_status", lambda _: {}),
    ("şarjım kaç kaldı?", "get_battery_status", lambda _: {}),
    ("pil durumu nedir?", "get_battery_status", lambda _: {}),
    ("batarya ne kadar kaldı?", "get_battery_status", lambda _: {}),
    ("şarj ne kadar?", "get_battery_status", lambda _: {}),
    ("pil yüzdesi kaç?", "get_battery_status", lambda _: {}),

    # 6. Uygulama Açma (Net & Pratik)
    ("{app} uygulamasını aç", "open_app", lambda a: {"app_name": a}),
    ("{app} aç", "open_app", lambda a: {"app_name": a}),
    ("{app}'ı aç", "open_app", lambda a: {"app_name": a}),
    ("{app} başlat", "open_app", lambda a: {"app_name": a}),
    ("{app}'a gir", "open_app", lambda a: {"app_name": a}),
]

apps_list = ["YouTube", "TikTok", "WhatsApp", "Instagram", "Spotify", "Kamera", "Galeri", "Ziraat", "Trendyol", "Twitter", "ChatGPT", "Notlar"]
times_list = ["06:30", "07:00", "07:30", "08:00", "08:15", "08:30", "09:00", "09:30", "10:00", "12:00", "14:00", "18:00", "20:00", "22:00"]
levels_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

dataset = []

# Generate Clean Direct Commands
for _ in range(300):
    for tmpl, tool, arg_fn in clean_templates:
        if "{level}" in tmpl:
            lvl = random.choice(levels_list)
            text = tmpl.format(level=lvl)
            args = arg_fn(lvl)
        elif "{time}" in tmpl:
            t = random.choice(times_list)
            text = tmpl.format(time=t)
            args = arg_fn(t)
        elif "{app}" in tmpl:
            a = random.choice(apps_list)
            text = tmpl.format(app=a)
            args = arg_fn(a)
        else:
            text = tmpl
            args = arg_fn(None)

        # Casing / Punctuation variations (realistic typing)
        variations = [text, text.lower(), text.capitalize(), text + ".", text + "!"]
        v_text = random.choice(variations)

        sample = {
            "messages": [
                {"role": "user", "content": v_text},
                {"role": "assistant", "content": f"<|tool_call|>{json.dumps({'name': tool, 'arguments': args}, ensure_ascii=False)}<|tool_call_end|>"}
            ]
        }
        dataset.append(sample)

# 7. Identity & Core Chat (Net & Profesyonel Asistan Kimliği)
chat_samples = [
    ("Sen kimsin?", "Ben Dorina! Talha tarafından geliştirilen, cihazını ve günlük işlerini yöneten yerel yapay zeka asistanınım."),
    ("sen kimsin", "Ben Dorina! Senin cihazında internetsiz ve yerel çalışan yapay zeka asistanınım Talha."),
    ("kimsin sen?", "Ben Dorina, kişisel yapay zeka asistanınım. Sana telefonunda ve komutlarında yardımcı olmak için buradayım."),
    ("Merhaba", "Merhaba Talha! Sana nasıl yardımcı olabilirim?"),
    ("Selam Dorina", "Selam Talha! Cihazında hazırım, ne yapmak istersin?"),
    ("Nasılsın?", "Çok iyiyim Talha, cihazında tam performans çalışıyorum! Sen nasılsın?"),
    ("Ne yapabilirsin?", "Ekran parlaklığını ve ses seviyesini ayarlayabilir, alarm kurabilir, el fenerini açıp kapatabilir, şarj durumunu söyleyebilir ve istediğin uygulamayı başlatabilirim."),
    ("Neler yapabiliyorsun?", "Senin için alarm kurabilir, parlaklık ve ses ayarı yapabilir, feneri açabilir ve YouTube, TikTok gibi uygulamalarını başlatabilirim."),
    ("Teşekkürler", "Rica ederim Talha! Her zaman buradayım."),
    ("Sağ ol Dorina", "Ne demek Talha, her zaman yanındayım!"),
]

for q, a in chat_samples:
    for _ in range(50): # Repeat to balance with tool calling
        dataset.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })

random.shuffle(dataset)
out_path = "data/dorina_gold_sft_v4/dorina_clean_direct_agent.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"[+] Toplam Temiz Örnek Sayısı: {len(dataset)}")
print(f"[+] Kaydedildi: {out_path}")
