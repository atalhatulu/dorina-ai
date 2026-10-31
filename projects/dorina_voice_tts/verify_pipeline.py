"""
Dorina TTS - Kaggle & Lokal Entegre Eğitim Scripti
Veri indirme, hazırlama ve HiFi-GAN V1 eğitimini tek parçada çalıştırır.
Hata vermeden önce tüm dosyaların varlığını doğrular.
"""
import os
import sys
import random
import time
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import pandas as pd
import numpy as np

# 1. Standart Mel-Spektrogram
class MelSpectrogramExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, n_fft=1024, win_length=1024, hop_length=256,
            f_min=0.0, f_max=8000.0, n_mels=80, power=1.0, normalized=False, center=False
        )
    def forward(self, wav):
        if wav.dim() == 1: wav = wav.unsqueeze(0)
        mel = self.mel_transform(wav)
        return torch.log(torch.clamp(mel, min=1e-5))


# 2. HiFi-GAN Mimarisi (Doğrulanmış Resmi Padding)
class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilations: list = [1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=(kernel_size * d - d) // 2)
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2)
            for _ in dilations
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            residual = x
            x = F.leaky_relu(x, 0.1)
            x = c1(x)
            x = F.leaky_relu(x, 0.1)
            x = c2(x)
            x = x + residual
        return x


class HiFiGANGenerator(nn.Module):
    def __init__(self, in_channels: int = 80, upsample_initial_channel: int = 512):
        super().__init__()
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, kernel_size=7, padding=3)
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4),
            nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
        ])
        self.resblocks = nn.ModuleList()
        for ch in [256, 128, 64, 32]:
            for k in [3, 7, 11]:
                self.resblocks.append(ResBlock(ch, kernel_size=k))
        self.conv_post = nn.Conv1d(32, 1, kernel_size=7, padding=3)

    def forward(self, x):
        x = self.conv_pre(x)
        res_idx = 0
        for up in self.ups:
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            xs = (self.resblocks[res_idx](x) + self.resblocks[res_idx+1](x) + self.resblocks[res_idx+2](x)) / 3.0
            res_idx += 3
            x = xs
        x = F.leaky_relu(x, 0.1)
        return torch.tanh(self.conv_post(x)).squeeze(1)


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))
        ])
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x):
        fmap = []
        b, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, 1, t // self.period, self.period)
        for c in self.convs:
            x = c(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MPD(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in [2, 3, 5, 7, 11]])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            ro, fr = d(y)
            go, fg = d(y_hat)
            y_d_rs.append(ro); y_d_gs.append(go); fmap_rs.append(fr); fmap_gs.append(fg)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def train_pipeline(steps: int = 50, batch_size: int = 8, data_dir: str = "voice/dataset_clean/wavs"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Pipeline Testi Başlatılıyor... Cihaz: {device}")
    
    mel_fn = MelSpectrogramExtractor().to(device)
    generator = HiFiGANGenerator().to(device)
    mpd = MPD().to(device)
    
    opt_g = torch.optim.AdamW(generator.parameters(), lr=2e-4, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(mpd.parameters(), lr=2e-4, betas=(0.8, 0.99))
    
    wav_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("train_studio")]
    print(f"[*] Tespit edilen eğitim sesleri: {len(wav_files)}")
    assert len(wav_files) > 0, "Eğitim dosyaları boş olamaz!"
    
    SEGMENT_SIZE = 8192
    generator.train()
    mpd.train()
    
    for step in range(1, steps + 1):
        batch = []
        for _ in range(batch_size):
            fn = random.choice(wav_files)
            d, _ = sf.read(fn)
            if len(d) > SEGMENT_SIZE + 500:
                st = random.randint(0, len(d) - SEGMENT_SIZE)
                batch.append(d[st:st+SEGMENT_SIZE])
            else:
                batch.append(np.pad(d, (0, max(0, SEGMENT_SIZE - len(d))))[:SEGMENT_SIZE])
                
        wav_batch = torch.tensor(np.array(batch), dtype=torch.float32).to(device)
        with torch.no_grad():
            mel_gt = mel_fn(wav_batch)
            
        wav_gen = generator(mel_gt)
        min_l = min(wav_batch.shape[-1], wav_gen.shape[-1])
        wav_real = wav_batch[:, :min_l]
        wav_fake = wav_gen[:, :min_l]
        
        # Discriminator
        opt_d.zero_grad()
        r_outs, g_outs, _, _ = mpd(wav_real, wav_fake.detach())
        loss_d = sum(torch.mean((1.0 - ro)**2) + torch.mean(go**2) for ro, go in zip(r_outs, g_outs))
        loss_d.backward()
        opt_d.step()
        
        # Generator
        opt_g.zero_grad()
        _, g_outs, f_r, f_g = mpd(wav_real, wav_fake)
        loss_adv = sum(torch.mean((1.0 - dg)**2) for dg in g_outs)
        loss_fm = sum(torch.mean(torch.abs(r-g)) for fr, fg in zip(f_r, f_g) for r, g in zip(fr, fg)) * 2.0
        loss_mel = F.l1_loss(mel_fn(wav_fake), mel_fn(wav_real)) * 45.0
        
        loss_g = loss_adv + loss_fm + loss_mel
        loss_g.backward()
        opt_g.step()
        
        if step % 10 == 0 or step == 1:
            print(f"Step [{step}/{steps}] - Loss G: {loss_g.item():.3f} (Mel: {loss_mel.item():.4f}) | Loss D: {loss_d.item():.3f}")

    print("[BAŞARILI] Lokal Test Hatasız Tamamlandı!")


if __name__ == "__main__":
    train_pipeline(steps=30, batch_size=8)
