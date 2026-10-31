"""
Dorina TTS - Ses İşleme ve Mel-Spektrogram Dönüşümleri (Audio DSP)
STFT ve Griffin-Lim (vokoder gerektirmeyen hızlı ses üretimi)
"""
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


class AudioConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0


class AudioProcessor(nn.Module):
    def __init__(self, config: AudioConfig = AudioConfig()):
        super().__init__()
        self.config = config
        
        # WAV -> Mel-Spektrogram
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            power=1.0,
            normalized=False
        )
        
        # Mel-Spektrogram -> WAV (Griffin-Lim)
        self.griffin_lim = T.GriffinLim(
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_iter=32,
            power=1.0
        )
        
        # Mel -> Lineer Spektrogram ters projeksiyon matrisi
        # (Griffin-Lim lineer STFT büyüklüğü bekler)
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=(config.n_fft // 2) + 1,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            sample_rate=config.sample_rate,
        )  # mel_fb shape: (freqs, mels) = (513, 80)
        # Pseudo-inverse matrisi: pinverse((513, 80)) -> shape (80, 513)
        # Ters projeksiyonda mel (b, 80, t)'den linear (b, 513, t)'ye gitmek için
        # (513, 80) x (80, t) yapmalıyız -> mel_fb.pinverse().T shape (513, 80)
        self.register_buffer("inv_mel_fb", torch.pinverse(mel_fb).t())

    def wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Girdi: (batch, time) veya (time,)
        Çıktı: Log-Mel Spectrogram (batch, n_mels, frames)
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            
        mel = self.mel_transform(wav)
        # Log ölçeği (numerik stabilite için 1e-5 eklenir)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel

    def mel_to_wav(self, log_mel: torch.Tensor) -> torch.Tensor:
        """
        Log-Mel Spektrogramdan tekrar duyulabilir WAV ses sinyali üretir.
        """
        if log_mel.dim() == 2:
            log_mel = log_mel.unsqueeze(0)
            
        # Log'dan lineer mel ölçeğine dön
        mel = torch.exp(log_mel)
        # inv_mel_fb shape: (freqs, mels)
        # mel shape: (batch, mels, frames)
        linear_spec = torch.einsum("fm, bmt -> bft", self.inv_mel_fb, mel)
        linear_spec = torch.clamp(linear_spec, min=0.0)
        
        # Griffin-Lim ile faz kestirimi ve dalga formu üretimi
        wav = self.griffin_lim(linear_spec)
        return wav.squeeze(0)


if __name__ == "__main__":
    processor = AudioProcessor()
    # 1 saniyelik sentetik ses testi (440Hz sinüs dalgası - La notası)
    t = torch.linspace(0, 1, 22050)
    fake_wav = torch.sin(2 * 3.14159 * 440 * t)
    
    mel = processor.wav_to_mel(fake_wav)
    print("Mel-Spektrogram Boyutu [Batch, Mels, Frames]:", mel.shape)
    
    reconstructed_wav = processor.mel_to_wav(mel)
    print("Reconstructed WAV Boyutu:", reconstructed_wav.shape)
    print("DSP Test Başarılı!")
