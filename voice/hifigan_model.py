"""
Dorina TTS - Sıfırdan HiFi-GAN V1 Nöral Vokoder Mimarisi (From Scratch)
Generator: Multi-Receptive Field (MRF) + Transposed Convolutions
Discriminators: Multi-Period Discriminator (MPD) + Multi-Scale Discriminator (MSD)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 1. GENERATOR (MRF & UPSAMPLING) ---

class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilations: list[int] = [1, 3, 5]):
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
    """Mel-Spektrogramı (80-bin) alıp 22.050 Hz analog ses dalgasına çeviren Generator"""
    def __init__(
        self,
        in_channels: int = 80,
        upsample_initial_channel: int = 512,
        upsample_rates: list[int] = [8, 8, 2, 2], # Toplam upsample = 8*8*2*2 = 256 (hop_length ile birebir eşit!)
        upsample_kernel_sizes: list[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: list[int] = [3, 7, 11]
    ):
        super().__init__()
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, kernel_size=7, padding=3)
        
        self.ups = nn.ModuleList()
        curr_ch = upsample_initial_channel
        for rate, k_size in zip(upsample_rates, upsample_kernel_sizes):
            self.ups.append(
                nn.ConvTranspose1d(
                    curr_ch,
                    curr_ch // 2,
                    kernel_size=k_size,
                    stride=rate,
                    padding=(k_size - rate) // 2
                )
            )
            curr_ch = curr_ch // 2
            
        self.resblocks = nn.ModuleList()
        # Her upsample katmanından sonra çoklu alıcı alan (MRF) blokları
        ch = upsample_initial_channel
        for _ in upsample_rates:
            ch = ch // 2
            for k in resblock_kernel_sizes:
                self.resblocks.append(ResBlock(ch, kernel_size=k))
                
        self.conv_post = nn.Conv1d(curr_ch, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # x: (batch, 80, frames)
        x = self.conv_pre(x)
        
        res_idx = 0
        for up in self.ups:
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            
            # MRF Blokları paralel toplanır
            xs = 0
            for _ in range(3): # 3 farklı kernel boyutu [3, 7, 11]
                xs = xs + self.resblocks[res_idx](x)
                res_idx += 1
            x = xs / 3.0
            
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x.squeeze(1) # (batch, time)


# --- 2. DISCRIMINATORS (MPD & MSD) ---

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


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in [2, 3, 5, 7, 11]
        ])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 128, 15, 1, padding=7),
            nn.Conv1d(128, 128, 41, 2, groups=4, padding=20),
            nn.Conv1d(128, 256, 41, 2, groups=16, padding=20),
            nn.Conv1d(256, 512, 41, 4, groups=16, padding=20),
            nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 5, 1, padding=2)
        ])
        self.conv_post = nn.Conv1d(1024, 1, 3, 1, padding=1)

    def forward(self, x):
        fmap = []
        x = x.unsqueeze(1)
        for c in self.convs:
            x = c(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator() for _ in range(3)
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y.unsqueeze(1)).squeeze(1)
                y_hat = self.meanpools[i - 1](y_hat.unsqueeze(1)).squeeze(1)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# --- 3. LOSS FONKSİYONLARI ---

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2.0


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
    return loss


def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        loss += l
    return loss


if __name__ == "__main__":
    gen = HiFiGANGenerator()
    params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(f"[*] HiFi-GAN Generator Parametre Sayısı: {params / 1e6:.2f} Milyon")
    
    # 80-mel test tensörü
    dummy_mel = torch.randn(2, 80, 100)
    fake_audio = gen(dummy_mel)
    print(f"[*] Giriş Mel: {dummy_mel.shape} -> Çıkış Ses Dalgası: {fake_audio.shape}")
    assert fake_audio.shape[-1] == 100 * 256, "Çıkış tam 256 katına (hop_length) yukarı örneklenmeli!"
    print("[OK] HiFi-GAN V1 Mimarisi Sıfırdan Kusursuz Çalışıyor!")
