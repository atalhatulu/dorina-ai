"""
Dorina TTS - Çıkarım (Inference) Betiği
Eğitilen modelden metin verip WAV ses dosyası üretir.
"""
import os
import torch
import soundfile as sf

from voice.text_processor import text_to_sequence, VOCAB
from voice.audio_dsp import AudioProcessor, AudioConfig
from voice.dorina_tts_model import DorinaTTSModel


class DorinaTTSInference:
    def __init__(self, checkpoint_path: str = "models/dorina_tts_prototype.pt", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.dsp = AudioProcessor().to(self.device)
        self.model = DorinaTTSModel(vocab_size=len(VOCAB)).to(self.device)
        
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[OK] Dorina TTS ağırlıkları yüklendi: {checkpoint_path}")
        else:
            print(f"[!] Uyarı: {checkpoint_path} bulunamadı, rastgele ağırlıklarla çalışıyor.")
            
        self.model.eval()

    @torch.no_grad()
    def speak(self, text: str, output_path: str = "output_dorina.wav") -> str:
        seq = torch.tensor([text_to_sequence(text)], dtype=torch.long, device=self.device)
        
        # Akustik modelden Mel-Spektrogram üretimi
        mel_out, dur_pred, pitch_pred, energy_pred = self.model(seq)
        
        # Audio DSP ile WAV ses sinyaline dönüştürme
        wav = self.dsp.mel_to_wav(mel_out.squeeze(0))
        wav_np = wav.cpu().numpy()
        
        sf.write(output_path, wav_np, samplerate=AudioConfig.sample_rate)
        print(f"[OK] Ses başarıyla üretildi: {output_path} ({len(wav_np) / AudioConfig.sample_rate:.2f} saniye)")
        return output_path


if __name__ == "__main__":
    tts = DorinaTTSInference()
    tts.speak("Merhaba Talha, ben Dorina. Kendi ses motorumla konuşuyorum!", "voice_output_test.wav")
