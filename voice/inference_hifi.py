"""
Dorina TTS - HiFi-GAN Nöral Vokoder Destekli Temiz Çıkarım (Inference)
Griffin-Lim'in metalik cızırtısını ve yırtık sesini ortadan kaldıran profesyonel ses motoru.
"""
import os
import torch
import soundfile as sf
from transformers import SpeechT5HifiGan

from voice.text_processor import text_to_sequence, VOCAB
from voice.audio_dsp import AudioConfig
from voice.dorina_tts_model import DorinaTTSModel


class DorinaHiFiTTS:
    def __init__(self, checkpoint_path: str = "models/dorina_tts_real.pt", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[*] Dorina HiFi-TTS başlatılıyor... Cihaz: {self.device}")
        
        # 1. Akustik Model (FastSpeech 2)
        self.model = DorinaTTSModel(vocab_size=len(VOCAB)).to(self.device)
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[OK] Akustik model yüklendi: {checkpoint_path}")
        self.model.eval()
        
        # 2. HiFi-GAN Nöral Vokoder (Mel -> Doğal Dalga Formu)
        print("[*] HiFi-GAN Vokoder yükleniyor...")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        self.vocoder.eval()
        print("[OK] Sistem pürüzsüz ses üretimi için hazır!")

    @torch.no_grad()
    def speak(self, text: str, output_path: str = "dorina_hifi_test.wav") -> str:
        # Metin -> Token ID
        seq = torch.tensor([text_to_sequence(text)], dtype=torch.long, device=self.device)
        
        # Akustik Modelden Mel-Spektrogram üret
        mel_out, dur_pred, pitch_pred, energy_pred = self.model(seq)
        
        # Mel boyutu: (1, 80, frames) -> HiFi-GAN bekler: (frames, 80)
        mel_t = mel_out.squeeze(0).transpose(0, 1)
        
        # Mel değerlerini HiFi-GAN dinamik aralığına uygun normalize et
        mel_norm = torch.clamp((mel_t + 5.0) / 5.0, min=-4.0, max=4.0)
        
        # Nöral Vokoder ile analog dalga formu üretimi
        waveform = self.vocoder(mel_norm)
        wav_np = waveform.cpu().numpy()
        
        # Sinyal tepe normalizasyonu (patlamaları engeller)
        max_val = max(abs(wav_np.max()), abs(wav_np.min()))
        if max_val > 0:
            wav_np = (wav_np / max_val) * 0.95
            
        # 16.000 Hz HiFi-GAN standart çıkışı
        sf.write(output_path, wav_np, samplerate=16000)
        print(f"[BAŞARILI] Pürüzsüz HiFi ses oluşturuldu: {output_path}")
        return output_path


if __name__ == "__main__":
    tts = DorinaHiFiTTS()
    tts.speak("Merhaba Talha, ben Dorina. Kendi sesimle konuşuyorum!", "dorina_hifi_test.wav")
