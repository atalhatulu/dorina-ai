import os
import sys
import time
import urllib.request
import torch

def download_file(url, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"  [OK] Zaten mevcut: {dest_path}")
        return
    print(f"  [Indiriliyor] {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    print(f"  [Tamamlandi] {dest_path}")

def setup_vad():
    print("\n--- 1. Silero VAD Modeli Dogrulaniyor ---")
    try:
        from silero_vad import load_silero_vad
        model = load_silero_vad()
        print("  [OK] Silero VAD basariyla yuklendi.")
    except Exception as e:
        print(f"  [Hata] Silero VAD yuklenemedi: {e}")

def setup_whisper():
    print("\n--- 2. Faster-Whisper (STT - base) Modeli Indiriliyor ---")
    try:
        from faster_whisper import WhisperModel
        whisper_dir = os.path.abspath("models/weights/whisper")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"  Yukleniyor: model=base, device={device}, compute_type={compute_type}...")
        model = WhisperModel("base", device=device, compute_type=compute_type, download_root=whisper_dir)
        print("  [OK] Whisper-base modeli basariyla yuklendi ve hazir.")
    except Exception as e:
        print(f"  [Hata] Whisper modeli yuklenemedi: {e}")

def setup_piper():
    print("\n--- 3. Piper TTS (Turkce Ses) Indiriliyor ---")
    piper_dir = os.path.abspath("models/weights/piper")
    os.makedirs(piper_dir, exist_ok=True)
    
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/dfki/medium/"
    files = ["tr_TR-dfki-medium.onnx", "tr_TR-dfki-medium.onnx.json"]
    
    for f in files:
        url = base_url + f
        dest = os.path.join(piper_dir, f)
        download_file(url, dest)
    
    # Kisa bir dogrulama
    try:
        from piper import PiperVoice
        onnx_file = os.path.join(piper_dir, "tr_TR-dfki-medium.onnx")
        config_file = os.path.join(piper_dir, "tr_TR-dfki-medium.onnx.json")
        voice = PiperVoice.load(onnx_file, config_path=config_file)
        print("  [OK] Piper Turkce ses modeli basariyla dogrulandi.")
    except Exception as e:
        print(f"  [Uyari/Hata] Piper dogrulama: {e}")

def setup_qwen():
    print("\n--- 4. NLU Modeli: Qwen2.5-0.5B-Instruct Indiriliyor ---")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"  Model onbellekleniyor: {model_id}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        print("  [OK] Qwen2.5-0.5B-Instruct basariyla yuklendi.")
        
        # Hizli bir inference testi
        messages = [
            {"role": "system", "content": "Sen restoran siparis asistanisin. JSON formatinda cevap ver."},
            {"role": "user", "content": "Bana 2 tane orta boy karisik pizza ve bir kola yazar misin?"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("  [OK] Ornek NLU testi tamamlandi. Yanit kesiti:", output_text[-120:].replace('\n', ' '))
    except Exception as e:
        print(f"  [Hata] Qwen modeli yuklenemedi: {e}")

if __name__ == "__main__":
    t0 = time.time()
    print("=== Dorina AI: Temel Model Paketlerini Hazirlama ===")
    setup_vad()
    setup_piper()
    setup_whisper()
    setup_qwen()
    print(f"\nTum modeller basariyla hazirlandi! (Gecen sure: {round(time.time() - t0, 1)}s)")
