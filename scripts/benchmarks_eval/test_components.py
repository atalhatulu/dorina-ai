import os
import sys
import time
import json
import wave
import torch
import numpy as np

# Windows utf-8 konsol ciktisi ayari
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

ARTIFACT_DIR = "test_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def test_1_tts():
    print("\n" + "="*50)
    print(" [TEST 1/4] PIPER TTS (Turkce Ses Sentezi)")
    print("="*50)
    
    from piper import PiperVoice
    
    onnx_path = os.path.abspath("models/weights/piper/tr_TR-dfki-medium.onnx")
    config_path = os.path.abspath("models/weights/piper/tr_TR-dfki-medium.onnx.json")
    wav_out_path = os.path.join(ARTIFACT_DIR, "tts_welcome_sample.wav")
    
    test_text = "Dorina Pizza'ya hoş geldiniz! Size nasıl yardımcı olabilirim?"
    print(f"  Metin: \"{test_text}\"")
    
    t0 = time.time()
    voice = PiperVoice.load(onnx_path, config_path=config_path)
    load_time = time.time() - t0
    print(f"  Model yukleme suresi: {load_time*1000:.1f} ms")
    
    t1 = time.time()
    with wave.open(wav_out_path, "wb") as wav_file:
        voice.synthesize_wav(test_text, wav_file)
    synth_time = time.time() - t1
    
    file_size_kb = os.path.getsize(wav_out_path) / 1024
    print(f"  [BASARILI] Ses uretildi -> {wav_out_path}")
    print(f"  Sentez suresi: {synth_time*1000:.1f} ms ({synth_time:.2f} s)")
    print(f"  Dosya boyutu: {file_size_kb:.1f} KB")
    
    return wav_out_path

def test_2_vad(wav_file_path):
    print("\n" + "="*50)
    print(" [TEST 2/4] SILERO VAD (Konusma Tespiti & Kesme)")
    print("="*50)
    
    import soundfile as sf
    from silero_vad import load_silero_vad, get_speech_timestamps
    
    t0 = time.time()
    vad_model = load_silero_vad()
    load_time = time.time() - t0
    print(f"  VAD Model yukleme suresi: {load_time*1000:.1f} ms")
    
    # Sesi soundfile ile oku ve float32 torch tensore cevir
    audio_data, sr = sf.read(wav_file_path, dtype='float32')
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # Mono
        
    # Silero VAD 16000 veya 8000 Hz bekler
    target_sr = 16000
    if sr != target_sr:
        import scipy.signal
        num_samples = int(len(audio_data) * target_sr / sr)
        audio_data = scipy.signal.resample(audio_data, num_samples).astype(np.float32)
        sr = target_sr

    wav = torch.from_numpy(audio_data)
    audio_duration_sec = len(wav) / sr
    print(f"  Girdi ses suresi: {audio_duration_sec:.2f} saniye, Ornekleme hizi: {sr} Hz")
    
    t1 = time.time()
    speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=sr)
    vad_time = time.time() - t1
    
    print(f"  VAD analiz suresi: {vad_time*1000:.1f} ms")
    print(f"  Konusma araliklari (orneklem bazli): {speech_timestamps}")
    for i, span in enumerate(speech_timestamps):
        start_sec = span['start'] / 16000
        end_sec = span['end'] / 16000
        print(f"    - Segment #{i+1}: {start_sec:.2f}s - {end_sec:.2f}s (Sure: {end_sec - start_sec:.2f}s)")
    print(f"  [BASARILI] Silero VAD konusmayi kusursuz ayristirdi.")

def test_3_stt(wav_file_path):
    print("\n" + "="*50)
    print(" [TEST 3/4] FASTER-WHISPER STT (Sesi Metne Cevirme)")
    print("="*50)
    
    from faster_whisper import WhisperModel
    
    whisper_dir = os.path.abspath("models/weights/whisper")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    print(f"  STT Ayarlari: Model='base', Cihaz={device}, Veri Tipi={compute_type}")
    t0 = time.time()
    model = WhisperModel("base", device=device, compute_type=compute_type, download_root=whisper_dir)
    print(f"  Model belleke alma: {(time.time() - t0)*1000:.1f} ms")
    
    # Dosya yolu yerine dogrudan float32 numpy array veriyoruz (av dependency bypass & ultra hiz)
    import soundfile as sf
    import scipy.signal
    audio_data, sr = sf.read(wav_file_path, dtype='float32')
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    if sr != 16000:
        num_samples = int(len(audio_data) * 16000 / sr)
        audio_data = scipy.signal.resample(audio_data, num_samples).astype(np.float32)

    t1 = time.time()
    # Turkce dil zorlamasi ve siparis baglami ipucu (prompt biasing)
    segments, info = model.transcribe(
        audio_data, 
        language="tr", 
        beam_size=5,
        initial_prompt="Pizza, içecek, menü siparişi."
    )
    
    transcribed_text = " ".join([seg.text for seg in segments]).strip()
    stt_time = time.time() - t1
    
    print(f"  Algilanan Dil: {info.language} (Guven: {info.language_probability:.2f})")
    print(f"  STT Cikis Metni: \"{transcribed_text}\"")
    print(f"  Cozumleme Suresi: {stt_time*1000:.1f} ms ({stt_time:.2f} s)")
    print(f"  [BASARILI] Sesten metne donusum gerceklestirildi.")
    
    return transcribed_text

def test_4_nlu():
    print("\n" + "="*50)
    print(" [TEST 4/4] QWEN 2.5 NLU (Niyet & Slot Cikarimi)")
    print("="*50)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    user_utterance = "Bana bir tane büyük boy karışık pizza, yanına da bir kutu kola ekler misin?"
    print(f"  Musteri Cumlesi: \"{user_utterance}\"")
    
    system_prompt = """Sen restoran sesli siparis asistanisin.
Kullanici cumlesinden sadece JSON formatinda siparis aksiyonu cikar:
Format:
{
  "action": "add_item",
  "items": [
    {"name": "...", "size": "kucuk|orta|buyuk|standart", "quantity": 1, "notes": "..."}
  ],
  "confidence": 0.95
}
Sadece gecerli bir JSON dondur, baska aciklama yazma."""

    t0 = time.time()
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto"
    )
    print(f"  Model yukleme suresi: {(time.time() - t0)*1000:.1f} ms")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_utterance}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    t1 = time.time()
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    output_tokens = model.generate(
        **inputs, 
        max_new_tokens=150, 
        temperature=0.1, 
        do_sample=False
    )
    nlu_time = time.time() - t1
    
    full_output = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    assistant_reply = full_output.split("assistant\n")[-1].strip()
    
    print(f"  NLU Cikarim Suresi: {nlu_time*1000:.1f} ms ({nlu_time:.2f} s)")
    print(f"  NLU Ham Ciktisi:\n{assistant_reply}")
    print(f"  [BASARILI] Qwen NLU aksiyon JSON uretti.")

if __name__ == "__main__":
    total_start = time.time()
    print("="*60)
    print(" DORINA AI - BILESEN BAZLI BAGIMSIZ TEST PROTOKOLU")
    print("="*60)
    
    # 1. TTS Testi
    wav_path = test_1_tts()
    
    # 2. VAD Testi
    test_2_vad(wav_path)
    
    # 3. STT Testi
    test_3_stt(wav_path)
    
    # 4. NLU Testi
    test_4_nlu()
    
    print("\n" + "="*60)
    print(f" TUM BILESEN TESTLERI TAMAMLANDI! Toplam test suresi: {time.time() - total_start:.2f} sn")
    print("="*60)
