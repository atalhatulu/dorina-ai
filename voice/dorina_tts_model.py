"""
Dorina TTS - FastSpeech 2 Akustik Model Mimarisi (From Scratch)
Non-autoregressive Transformer tabanlı, Duration, Pitch ve Energy kestiricili mimari.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTBlock(nn.Module):
    """Feed-Forward Transformer Blok (Self-Attention + 1D Conv)"""
    def __init__(self, d_model: int, n_heads: int, conv_filter_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # FastSpeech 1D Conv Katmanları (Pozisyonel bilgiyi ve lokal bağlamı mükemmel yakalar)
        self.conv1 = nn.Conv1d(d_model, conv_filter_size, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(conv_filter_size, d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-Head Self-Attention
        residual = x
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(residual + self.dropout(attn_out))
        
        # 1D Convolutional FFN
        residual = x
        out = F.relu(self.conv1(x.transpose(1, 2)))
        out = self.conv2(out).transpose(1, 2)
        x = self.norm2(residual + self.dropout(out))
        return x


class VariancePredictor(nn.Module):
    """Duration, Pitch veya Energy kestirimi yapan konvolüsyonel tahminci"""
    def __init__(self, d_model: int, filter_size: int = 256, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, filter_size, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.LayerNorm(filter_size)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm(filter_size)
        self.linear = nn.Linear(filter_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        out = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        out = F.relu(self.norm1(out))
        out = self.dropout(out)
        
        out = self.conv2(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(self.norm2(out))
        out = self.dropout(out)
        
        out = self.linear(out)  # (batch, seq_len, 1)
        return out.squeeze(-1)


class LengthRegulator(nn.Module):
    """
    Her fonem/harf vektörünü, Duration Predictor'ın belirlediği süre (frame) kadar tekrarlayarak
    metin boyutundan ses (frame) boyutuna genişletir.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, durations):
        # x: (batch, seq_len, d_model), durations: (batch, seq_len)
        output = []
        for batch_idx in range(x.size(0)):
            seq = x[batch_idx]
            durs = durations[batch_idx]
            expanded = []
            for token_vec, dur in zip(seq, durs):
                dur_int = max(1, int(round(dur.item())))
                expanded.append(token_vec.repeat(dur_int, 1))
            output.append(torch.cat(expanded, dim=0))
            
        # Pad to max length in batch
        max_len = max(o.size(0) for o in output)
        padded_output = []
        for o in output:
            pad_len = max_len - o.size(0)
            if pad_len > 0:
                padded = F.pad(o, (0, 0, 0, pad_len))
            else:
                padded = o
            padded_output.append(padded.unsqueeze(0))
        return torch.cat(padded_output, dim=0)


class DorinaTTSModel(nn.Module):
    """
    Dorina TTS - FastSpeech 2 Çekirdek Akustik Modeli
    """
    def __init__(
        self,
        vocab_size: int = 42,
        d_model: int = 192,
        n_heads: int = 2,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        conv_filter_size: int = 384,
        n_mels: int = 80,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Encoder (Harfleri anlar)
        self.encoder_layers = nn.ModuleList([
            FFTBlock(d_model, n_heads, conv_filter_size, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Variance Adaptor (Süre, Tonlama, Vurgu)
        self.duration_predictor = VariancePredictor(d_model)
        self.pitch_predictor = VariancePredictor(d_model)
        self.energy_predictor = VariancePredictor(d_model)
        
        self.pitch_embedding = nn.Conv1d(1, d_model, kernel_size=9, padding=4)
        self.energy_embedding = nn.Conv1d(1, d_model, kernel_size=9, padding=4)
        self.length_regulator = LengthRegulator()
        
        # Decoder (Genişletilmiş temsili Mel-Spektrogram çerçevelerine dönüştürür)
        self.decoder_layers = nn.ModuleList([
            FFTBlock(d_model, n_heads, conv_filter_size, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Mel Projeksiyon (d_model -> 80 mel kanalı)
        self.mel_linear = nn.Linear(d_model, n_mels)

    def forward(self, text_seq, target_durations=None, target_pitch=None, target_energy=None):
        # 1. Embedding
        x = self.embedding(text_seq)
        
        # 2. Encoder
        for layer in self.encoder_layers:
            x = layer(x)
            
        # 3. Predict Duration, Pitch, Energy
        dur_pred = self.duration_predictor(x)
        pitch_pred = self.pitch_predictor(x)
        energy_pred = self.energy_predictor(x)
        
        # Eğitimde ground-truth, inferansta tahmin edilen değerler kullanılır
        if target_durations is not None:
            dur_to_use = target_durations
        else:
            # İnferansta harf başına mantıklı bir süre (frame: 2-6 frame arası, ~25-70ms)
            dur_to_use = torch.clamp(torch.round(F.softplus(dur_pred) * 3 + 1), min=1, max=12)
            
        pitch_to_use = target_pitch if target_pitch is not None else pitch_pred
        energy_to_use = target_energy if target_energy is not None else energy_pred
        
        # Variance embedding ekleme
        p_emb = self.pitch_embedding(pitch_to_use.unsqueeze(1)).transpose(1, 2)
        e_emb = self.energy_embedding(energy_to_use.unsqueeze(1)).transpose(1, 2)
        x = x + p_emb + e_emb
        
        # 4. Length Regulation (Harften zamana genişleme)
        x = self.length_regulator(x, dur_to_use)
        
        # 5. Decoder
        for layer in self.decoder_layers:
            x = layer(x)
            
        # 6. Mel Spectrogram Tahmini
        mel_out = self.mel_linear(x)  # (batch, frames, n_mels)
        mel_out = mel_out.transpose(1, 2)  # (batch, n_mels, frames)
        
        return mel_out, dur_pred, pitch_pred, energy_pred


if __name__ == "__main__":
    model = DorinaTTSModel()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Dorina TTS Model Parametre Sayısı: {param_count / 1e6:.2f} Milyon")
    
    # Dummy forward pass testi
    fake_text = torch.tensor([[2, 20, 10, 25, 3]])  # 5 token
    fake_dur = torch.tensor([[3, 4, 3, 5, 2]])      # Toplam 17 frame
    fake_pitch = torch.randn(1, 5)
    fake_energy = torch.randn(1, 5)
    
    mel, d, p, e = model(fake_text, fake_dur, fake_pitch, fake_energy)
    print("Üretilen Mel-Spektrogram Boyutu [Batch, Mels, Frames]:", mel.shape)
    print("Model Mimarisi Kusursuz Çalışıyor!")
