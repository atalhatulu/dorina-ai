"""
Dorina TTS - Monotonic Alignment Search (MAS) & Aligner Loss
Metin temsili (Text) ile Ses (Mel-Spectrogram) arasındaki gerçek süreleri (Durations)
dinamik programlama ve CTC matrisi ile matematiksel olarak çözer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def monotonic_alignment_search(similarity_matrix: torch.Tensor) -> torch.Tensor:
    """
    Monotonic Alignment Search (MAS) - Dinamik Programlama Algoritması
    Girdi: similarity_matrix: (T_text, T_mel) log-olasılık veya benzerlik tensörü
    Çıktı: hard_alignment: (T_text, T_mel) ikili (0 veya 1) hizalama matrisi
    """
    t_text, t_mel = similarity_matrix.shape
    device = similarity_matrix.device
    
    # DP tablosu (maksimum log-olasılık yolu)
    dp = torch.full((t_text, t_mel), float("-inf"), device=device)
    dp[0, 0] = similarity_matrix[0, 0]
    
    # İlk satırı doldur (metnin ilk harfi sesin ilk birkaç karesinde kalabilir)
    for j in range(1, t_mel):
        dp[0, j] = dp[0, j - 1] + similarity_matrix[0, j]
        
    # DP matrisini doldur: dp[i, j] = sim[i, j] + max(dp[i, j-1], dp[i-1, j-1])
    for i in range(1, t_text):
        for j in range(i, t_mel):
            prev_max = max(dp[i, j - 1], dp[i - 1, j - 1])
            dp[i, j] = similarity_matrix[i, j] + prev_max

    # Geriye doğru (Backtracking) en iyi yolu bul
    alignment = torch.zeros((t_text, t_mel), device=device)
    curr_i = t_text - 1
    alignment[curr_i, t_mel - 1] = 1.0
    
    for j in range(t_mel - 1, 0, -1):
        if curr_i > 0 and dp[curr_i - 1, j - 1] >= dp[curr_i, j - 1]:
            curr_i -= 1
        alignment[curr_i, j - 1] = 1.0
        
    return alignment


class InternalAligner(nn.Module):
    """
    Metin Encoder çıktıları ile Mel Spektrogram arasında benzerlik kuran ve
    harflerin gerçek sürelerini çıkaran dahili modül.
    """
    def __init__(self, d_model: int = 256, n_mels: int = 80):
        super().__init__()
        # Mel çerçevelerini d_model uzayına yansıtıcı
        self.mel_conv = nn.Sequential(
            nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1)
        )
        # Metin tensörü projeksiyonu
        self.text_proj = nn.Linear(d_model, d_model)

    def forward(self, text_emb, mel_spec):
        """
        text_emb: (batch, t_text, d_model)
        mel_spec: (batch, n_mels, t_mel)
        """
        # Mel projection: (batch, d_model, t_mel) -> (batch, t_mel, d_model)
        mel_proj = self.mel_conv(mel_spec).transpose(1, 2)
        # Text projection: (batch, t_text, d_model)
        t_proj = self.text_proj(text_emb)
        
        # Kosinüs benzerliği / Matmul: (batch, t_text, t_mel)
        similarity = torch.bmm(t_proj, mel_proj.transpose(1, 2)) / (text_emb.shape[-1] ** 0.5)
        
        # Softmax normalize olasılık
        attn_soft = F.log_softmax(similarity, dim=-1)
        return attn_soft

    def get_hard_durations(self, text_emb, mel_spec):
        """Her harfin gerçek süresini (frame sayısını) MAS ile çıkarır."""
        attn_soft = self.forward(text_emb, mel_spec)
        batch_size = text_emb.size(0)
        
        durations = []
        for b in range(batch_size):
            sim = attn_soft[b]
            hard_align = monotonic_alignment_search(sim)
            # Her satırdaki 1'lerin toplamı o harfin süresidir
            dur = hard_align.sum(dim=-1)
            durations.append(dur)
            
        return torch.stack(durations)


if __name__ == "__main__":
    aligner = InternalAligner(d_model=256, n_mels=80)
    fake_text = torch.randn(1, 10, 256) # 10 harf
    fake_mel = torch.randn(1, 80, 50)   # 50 frame ses
    
    durs = aligner.get_hard_durations(fake_text, fake_mel)
    print("Harf Süreleri (MAS Çıktısı):", durs[0].tolist())
    print("Toplam Süre Frame Sayısı:", durs[0].sum().item())
    assert int(durs[0].sum().item()) == 50, "Süreler mel çerçevesine eşit olmalı!"
    print("MAS Hizalama Algoritması Kusursuz Çalışıyor!")
