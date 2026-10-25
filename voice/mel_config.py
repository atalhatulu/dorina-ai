"""
Dorina TTS - Tek Standart Mel ve Ses Konfigürasyonu (Single Source of Truth)
Tüm modeller (HiFi-GAN Vokoder, FastPitch Akustik Model ve Çıkarım) bu dosyayı kullanır.
"""
import torch
import torch.nn as nn
import torchaudio


class MelConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0


class MelSpectrogramExtractor(nn.Module):
    def __init__(self, config: MelConfig = MelConfig()):
        super().__init__()
        self.config = config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            power=1.0,
            normalized=False,
            center=False
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (batch, time) veya (time,)
        Çıktı: HiFi-GAN standardı doğal logaritma Mel-spektrogramı: (batch, n_mels, frames)
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            
        mel = self.mel_transform(wav)
        # Doğal logaritma (ln) ve 1e-5 clamp (Claude ve HiFi-GAN resmi standardı)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel


# Global tek mel fonksiyonu
GLOBAL_MEL_EXTRACTOR = None

def get_mel_spectrogram(wav: torch.Tensor) -> torch.Tensor:
    global GLOBAL_MEL_EXTRACTOR
    if GLOBAL_MEL_EXTRACTOR is None:
        GLOBAL_MEL_EXTRACTOR = MelSpectrogramExtractor()
    device = wav.device
    if next(GLOBAL_MEL_EXTRACTOR.parameters(), None) is not None:
        GLOBAL_MEL_EXTRACTOR = GLOBAL_MEL_EXTRACTOR.to(device)
    return GLOBAL_MEL_EXTRACTOR(wav)


if __name__ == "__main__":
    t = torch.randn(1, 22050)
    mel = get_mel_spectrogram(t)
    print("Mel Config Test Başarılı! Shape:", mel.shape)
    print("Min / Max:", mel.min().item(), mel.max().item())
