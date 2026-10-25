"""
Dorina FastPitch 2 - Claude Tavsiyeli Ölçeklendirilmiş Mimari
d_model = 256, 4 FFT Encoder, 4 FFT Decoder, Dahili MAS Aligner
Toplam: ~16M Parametre (Sweet spot)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from voice.alignment import InternalAligner


class FFTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, conv_filter_size: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, conv_filter_size, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(conv_filter_size, d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(residual + self.dropout(attn_out))
        residual = x
        out = F.relu(self.conv1(x.transpose(1, 2)))
        out = self.conv2(out).transpose(1, 2)
        return self.norm2(residual + self.dropout(out))


class VariancePredictor(nn.Module):
    def __init__(self, d_model: int = 256, filter_size: int = 256, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, filter_size, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.LayerNorm(filter_size)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm(filter_size)
        self.linear = nn.Linear(filter_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x.transpose(1, 2)).transpose(1, 2)))
        out = self.dropout(out)
        out = F.relu(self.norm2(self.conv2(out.transpose(1, 2)).transpose(1, 2)))
        out = self.dropout(out)
        return self.linear(out).squeeze(-1)


class LengthRegulator(nn.Module):
    def forward(self, x, durations):
        output = []
        for batch_idx in range(x.size(0)):
            seq = x[batch_idx]
            durs = durations[batch_idx]
            expanded = []
            for token_vec, dur in zip(seq, durs):
                dur_int = max(1, int(round(dur.item())))
                expanded.append(token_vec.repeat(dur_int, 1))
            output.append(torch.cat(expanded, dim=0))
            
        max_len = max(o.size(0) for o in output)
        padded = [F.pad(o, (0, 0, 0, max_len - o.size(0))) if max_len > o.size(0) else o for o in output]
        return torch.stack(padded)


class DorinaFastPitch(nn.Module):
    def __init__(
        self,
        vocab_size: int = 46,
        d_model: int = 256,
        n_heads: int = 4,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        conv_filter_size: int = 512,
        n_mels: int = 80,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 1. Metin Kodlayıcı (Encoder)
        self.encoder = nn.ModuleList([
            FFTBlock(d_model, n_heads, conv_filter_size, dropout)
            for _ in range(encoder_layers)
        ])
        
        # 2. Dahili MAS Hizalayıcı (Aligner)
        self.aligner = InternalAligner(d_model=d_model, n_mels=n_mels)
        
        # 3. Süre ve Tonlama Kestiricileri
        self.duration_predictor = VariancePredictor(d_model)
        self.pitch_predictor = VariancePredictor(d_model)
        self.energy_predictor = VariancePredictor(d_model)
        
        self.pitch_embedding = nn.Conv1d(1, d_model, kernel_size=9, padding=4)
        self.energy_embedding = nn.Conv1d(1, d_model, kernel_size=9, padding=4)
        self.length_regulator = LengthRegulator()
        
        # 4. Mel Kod Çözücü (Decoder)
        self.decoder = nn.ModuleList([
            FFTBlock(d_model, n_heads, conv_filter_size, dropout)
            for _ in range(decoder_layers)
        ])
        
        self.mel_linear = nn.Linear(d_model, n_mels)

    def forward(self, text_seq, mel_target=None):
        # Text embedding
        x = self.embedding(text_seq)
        for layer in self.encoder:
            x = layer(x)
            
        dur_pred = self.duration_predictor(x)
        pitch_pred = self.pitch_predictor(x)
        energy_pred = self.energy_predictor(x)
        
        # Eğitimde gerçek süreleri Aligner (MAS) ile çıkarır
        if mel_target is not None:
            with torch.no_grad():
                dur_to_use = self.aligner.get_hard_durations(x, mel_target)
        else:
            # İnferansta tahmin edilen süreyi log'dan çevir
            dur_to_use = torch.clamp(torch.round(torch.exp(dur_pred) - 1.0), min=1, max=25)
            
        p_emb = self.pitch_embedding(pitch_pred.unsqueeze(1)).transpose(1, 2)
        e_emb = self.energy_embedding(energy_pred.unsqueeze(1)).transpose(1, 2)
        x = x + p_emb + e_emb
        
        x = self.length_regulator(x, dur_to_use)
        
        for layer in self.decoder:
            x = layer(x)
            
        mel_out = self.mel_linear(x).transpose(1, 2)
        return mel_out, dur_pred, dur_to_use, pitch_pred, energy_pred


if __name__ == "__main__":
    m = DorinaFastPitch()
    params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"[*] Dorina FastPitch Parametre Sayısı: {params / 1e6:.2f} Milyon")
