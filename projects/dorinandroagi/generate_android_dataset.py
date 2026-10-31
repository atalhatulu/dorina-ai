import json
import random
import os
from tokenizers import Tokenizer

def generate_android_agent_dataset():
    print("=" * 70)
    print("DORINA-ANDROID AGENT VERI SETI URETICI (15 TEMEL SISTEM AKSIYONU)")
    print("=" * 70)

    # 15 Temel Android Eylemi ve Dogal Turkce/Ingilizce Varyasyon Kaliplari
    actions = [
        # 1. ALARM
        {
            "tool": "set_alarm",
            "patterns": [
                ("Yarın sabah saat {time} için alarm kur", {"time": "{time}", "label": "Alarm"}),
                ("Saat {time}'e alarm kur", {"time": "{time}", "label": "Alarm"}),
                ("Beni saat {time}'de uyandır", {"time": "{time}", "label": "Uyan"}),
                ("Set an alarm for {time}", {"time": "{time}", "label": "Alarm"}),
                ("Wake me up at {time}", {"time": "{time}", "label": "Wake up"})
            ],
            "params": {"time": ["06:30", "07:00", "07:45", "08:00", "08:30", "09:00", "12:00", "14:30", "22:00"]}
        },
        # 2. FLASHLIGHT (FENER)
        {
            "tool": "toggle_flashlight",
            "patterns": [
                ("Feneri aç", {"state": True}),
                ("Telefonun flaşını yak", {"state": True}),
                ("Karanlık oldu feneri açar mısın", {"state": True}),
                ("Turn on the flashlight", {"state": True}),
                ("Feneri kapat", {"state": False}),
                ("Flaşı söndür", {"state": False}),
                ("Turn off flashlight", {"state": False})
            ],
            "params": {}
        },
        # 3. SES SEVIYESI (VOLUME)
        {
            "tool": "set_volume",
            "patterns": [
                ("Telefonun sesini %{val} yap", {"level": "{val}"}),
                ("Sesi %{val} seviyesine getir", {"level": "{val}"}),
                ("Medya sesini {val} yap", {"level": "{val}"}),
                ("Set volume to {val}%", {"level": "{val}"})
            ],
            "params": {"val": [0, 10, 25, 50, 70, 80, 100]}
        },
        # 4. PARLAKLIK (BRIGHTNESS)
        {
            "tool": "set_brightness",
            "patterns": [
                ("Ekran parlaklığını %{val} yap", {"level": "{val}"}),
                ("Parlaklığı kıs %{val} olsun", {"level": "{val}"}),
                ("Ekranı aydınlat %{val} yap", {"level": "{val}"}),
                ("Set screen brightness to {val}%", {"level": "{val}"})
            ],
            "params": {"val": [10, 20, 50, 75, 100]}
        },
        # 5. WI-FI
        {
            "tool": "toggle_wifi",
            "patterns": [
                ("Wi-Fi'ı aç", {"state": True}),
                ("Kablosuz ağı bağla", {"state": True}),
                ("Turn on Wi-Fi", {"state": True}),
                ("Wi-Fi'ı kapat", {"state": False}),
                ("İnterneti kes Wi-Fi kapat", {"state": False}),
                ("Turn off Wi-Fi", {"state": False})
            ],
            "params": {}
        },
        # 6. BLUETOOTH
        {
            "tool": "toggle_bluetooth",
            "patterns": [
                ("Bluetooth'u aç", {"state": True}),
                ("Kulaklığa bağlanmak için bluetooth aç", {"state": True}),
                ("Turn on bluetooth", {"state": True}),
                ("Bluetooth'u kapat", {"state": False}),
                ("Turn off bluetooth", {"state": False})
            ],
            "params": {}
        },
        # 7. BATARYA / PIL
        {
            "tool": "get_battery_status",
            "patterns": [
                ("Şarjım kaç kaldı?", {}),
                ("Batarya durumu nedir?", {}),
                ("Telefonun şarjı ne kadar?", {}),
                ("Pil yüzdesini söyle", {}),
                ("What is my battery level?", {}),
                ("How much battery do I have?", {})
            ],
            "params": {}
        },
        # 8. UYGULAMA AC (OPEN APP)
        {
            "tool": "open_app",
            "patterns": [
                ("{app} uygulamasını aç", {"app_name": "{app}"}),
                ("{app}'ı başlat", {"app_name": "{app}"}),
                ("Bana {app}'ı açar mısın", {"app_name": "{app}"}),
                ("Open {app}", {"app_name": "{app}"}),
                ("Launch {app}", {"app_name": "{app}"})
            ],
            "params": {"app": ["WhatsApp", "Spotify", "YouTube", "Instagram", "Kamera", "Haritalar", "Galeri", "Chrome", "Twitter"]}
        },
        # 9. WHATSAPP MESAJ
        {
            "tool": "send_whatsapp",
            "patterns": [
                ("{person} kişisine WhatsApp'tan '{msg}' yaz", {"contact": "{person}", "message": "{msg}"}),
                ("WhatsApp üzerinden {person}'e '{msg}' mesajını gönder", {"contact": "{person}", "message": "{msg}"}),
                ("Send WhatsApp message to {person}: '{msg}'", {"contact": "{person}", "message": "{msg}"})
            ],
            "params": {
                "person": ["Ahmet", "Mehmet", "Ayşe", "Fatma", "Annem", "Babam", "Kardeşim", "Ali"],
                "msg": ["Geliyorum", "Neredesin?", "Tamamdır", "Toplantıdayım", "Görüşürüz", "Arayacağım"]
            }
        },
        # 10. ARAMA YAP (CALL)
        {
            "tool": "make_phone_call",
            "patterns": [
                ("{person} kişisini ara", {"contact": "{person}"}),
                ("{person}'e telefon et", {"contact": "{person}"}),
                ("Rehberden {person}'i ara", {"contact": "{person}"}),
                ("Call {person}", {"contact": "{person}"})
            ],
            "params": {"person": ["Annem", "Babam", "Ahmet", "Mehmet", "Ayşe", "Doktor", "Usta"]}
        },
        # 11. MUZIK KONTROL
        {
            "tool": "media_control",
            "patterns": [
                ("Müziği durdur", {"action": "pause"}),
                ("Şarkıyı duraklat", {"action": "pause"}),
                ("Müziği çalmaya devam et", {"action": "play"}),
                ("Sıradaki şarkıya geç", {"action": "next"}),
                ("Önceki parçaya dön", {"action": "previous"}),
                ("Pause the music", {"action": "pause"}),
                ("Skip to next song", {"action": "next"})
            ],
            "params": {}
        },
        # 12. HAVA DURUMU
        {
            "tool": "get_weather",
            "patterns": [
                ("{city} hava durumu nasıl?", {"city": "{city}"}),
                ("{city}'da bugün yağmur var mı?", {"city": "{city}"}),
                ("What is the weather in {city}?", {"city": "{city}"})
            ],
            "params": {"city": ["Ankara", "İstanbul", "İzmir", "Bursa", "Antalya", "Trabzon", "London", "Berlin"]}
        },
        # 13. WEB ARAMA
        {
            "tool": "search_web",
            "patterns": [
                ("İnternette ara: {q}", {"query": "{q}"}),
                ("Google'da {q} konusunu araştır", {"query": "{q}"}),
                ("Search web for {q}", {"query": "{q}"})
            ],
            "params": {"q": ["18. osmanlı padişahı", "yapay zeka haberleri", "dolar kuru", "en yakın nöbetçi eczane", "python hızlı sıralama algoritması"]}
        }
    ]

    dataset = []

    # Eylemleri varyasyonlarla uret
    for act in actions:
        tool_name = act["tool"]
        patterns = act["patterns"]
        params = act["params"]

        for pat_text, pat_args in patterns:
            # Parametre kombinasyonlari
            if not params:
                for _ in range(80):
                    call_json = json.dumps({"name": tool_name, "arguments": pat_args}, ensure_ascii=False)
                    dataset.append({
                        "messages": [
                            {"role": "user", "content": pat_text},
                            {"role": "assistant", "content": f"<|tool_call|>{call_json}<|tool_call_end|>"}
                        ]
                    })
            else:
                p_keys = list(params.keys())
                if len(p_keys) == 1:
                    k = p_keys[0]
                    for val in params[k]:
                        txt = pat_text.replace(f"{{{k}}}", str(val))
                        args = {arg_k: (str(val) if arg_v == f"{{{k}}}" else arg_v) for arg_k, arg_v in pat_args.items()}
                        for _ in range(12):
                            call_json = json.dumps({"name": tool_name, "arguments": args}, ensure_ascii=False)
                            dataset.append({
                                "messages": [
                                    {"role": "user", "content": txt},
                                    {"role": "assistant", "content": f"<|tool_call|>{call_json}<|tool_call_end|>"}
                                ]
                            })
                elif len(p_keys) == 2:
                    k1, k2 = p_keys[0], p_keys[1]
                    for val1 in params[k1]:
                        for val2 in params[k2]:
                            txt = pat_text.replace(f"{{{k1}}}", str(val1)).replace(f"{{{k2}}}", str(val2))
                            args = {}
                            for arg_k, arg_v in pat_args.items():
                                if arg_v == f"{{{k1}}}": args[arg_k] = str(val1)
                                elif arg_v == f"{{{k2}}}": args[arg_k] = str(val2)
                                else: args[arg_k] = arg_v
                            call_json = json.dumps({"name": tool_name, "arguments": args}, ensure_ascii=False)
                            dataset.append({
                                "messages": [
                                    {"role": "user", "content": txt},
                                    {"role": "assistant", "content": f"<|tool_call|>{call_json}<|tool_call_end|>"}
                                ]
                            })

    # Negatif Ornekler (%25 - Arac cagirmamasi gereken normal sohbet)
    negatives = [
        ("Merhaba nasılsın?", "Merhaba Talha! Gayet iyiyim, telefonunda sana nasıl yardımcı olabilirim?"),
        ("Sen kimsin?", "Ben Dorina, telefonunda yerel ve internetsiz çalışan kişisel yapay zeka asistanınım."),
        ("Teşekkürler", "Rica ederim, her zaman buradayım!"),
        ("Görüşürüz", "Görüşmek üzere, iyi günler!"),
        ("Bugün nasılsın?", "Harikayım! Senin için yapabileceğim bir işlem var mı?"),
        ("What can you do?", "I can set alarms, control phone settings like volume and flashlight, send messages, and search the web."),
        ("Python'da liste nedir?", "Python'da listeler, birden çok veriyi sıralı ve değiştirilebilir şekilde tutan veri yapılarıdır.")
    ]
    for q, a in negatives:
        for _ in range(120):
            dataset.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })

    random.seed(42)
    random.shuffle(dataset)

    tok_path = "tokenizers/dorina_tokenizer_v4_sft.json"
    tok = Tokenizer.from_file(tok_path)
    
    # Tokenize & Masking
    clean_samples = []
    for item in dataset:
        messages = item["messages"]
        input_ids = []
        labels = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            header = tok.encode(f"<|im_start|>{role}\n").ids
            body = tok.encode(content).ids
            footer = tok.encode("<|im_end|>\n").ids

            block = header + body + footer
            input_ids.extend(block)
            if role == "assistant":
                labels.extend([-100] * len(header))
                labels.extend(body)
                labels.extend(footer)
            else:
                labels.extend([-100] * len(block))

        input_ids = input_ids[:256]
        labels = labels[:256]
        if any(lbl != -100 for lbl in labels):
            clean_samples.append((input_ids, labels))

    out_file = "data/dorina_gold_sft_v4/dorina_android_agent_dataset.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(clean_samples, f)

    print(f"\n[+] DORINA-ANDROID AGENT VERI SETI TAMAMLANDI!")
    print(f"  - Toplam Ornek Sayisi: {len(clean_samples):,}")
    print(f"  - Kaydedilen Dosya: {out_file}")

if __name__ == "__main__":
    generate_android_agent_dataset()
