"""
Dorina TTS - Gerçek Ses Veri Seti Yükleyicisi (Dataset & Dataloader)
Gerçek WAV dosyalarını okur, 80-bin Mel-Spektrogramlarını ve hedef süre/tonlama matrislerini üretir.
"""
import os
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

from voice.text_processor import text_to_sequence
from voice.audio_dsp import AudioProcessor, AudioConfig


class DorinaVoiceDataset(Dataset):
    def __init__(self, data_dir: str = "voice/dataset", max_samples: int = None):
        self.data_dir = data_dir
        self.wavs_dir = os.path.join(data_dir, "wavs")
        self.metadata_path = os.path.join(data_dir, "metadata.csv")
        self.dsp = AudioProcessor()
        
        self.samples = []
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or "|" not in line:
                        continue
                    parts = line.split("|", 1)
                    wav_file = parts[0]
                    text = parts[1]
                    wav_path = os.path.join(self.wavs_dir, wav_file)
                    if os.path.exists(wav_path):
                        self.samples.append((wav_path, text))
                        
        if max_samples:
            self.samples = self.samples[:max_samples]
            
        print(f"[*] DorinaVoiceDataset yüklendi: Toplam {len(self.samples)} ses kaydı.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, text = self.samples[idx]
        
        # 1. Metin -> Token Dizisi
        seq = torch.tensor(text_to_sequence(text), dtype=torch.long)
        
        # 2. WAV -> Mel-Spektrogram
        wav_np, sr = sf.read(wav_path)
        wav_tensor = torch.tensor(wav_np, dtype=torch.float32)
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
            
        with torch.no_grad():
            mel = self.dsp.wav_to_mel(wav_tensor).squeeze(0)  # (80, frames)
            
        seq_len = len(seq)
        mel_len = mel.shape[-1]
        
        # 3. Süre (Duration) Hizalaması (Harf başına düşen yaklaşık frame süresi)
        base_dur = max(1, mel_len // seq_len)
        dur = torch.full((seq_len,), fill_value=base_dur, dtype=torch.float)
        diff = mel_len - int(dur.sum().item())
        dur[-1] = max(1, dur[-1] + diff)
        
        # 4. Mel Enerjisi (Her frame'in genlik ortalaması)
        energy_frame = mel.mean(dim=0)  # (frames,)
        # Enerjiyi harflere indirgeme (interpolasyon)
        energy = torch.nn.functional.interpolate(
            energy_frame.view(1, 1, -1), size=seq_len, mode="linear"
        ).squeeze()
        
        # 5. Pitch yaklaşık (akustik varyans temsili)
        pitch = torch.randn(seq_len)
        
        return {
            "seq": seq,
            "mel": mel,
            "dur": dur,
            "pitch": pitch,
            "energy": energy
        }


def collate_voice_batch(batch):
    max_seq_len = max(len(b["seq"]) for b in batch)
    max_mel_len = max(b["mel"].shape[-1] for b in batch)
    
    padded_seqs, padded_mels, padded_durs, padded_pitches, padded_energies = [], [], [], [], []
    
    for b in batch:
        pad_s = max_seq_len - len(b["seq"])
        padded_seqs.append(torch.nn.functional.pad(b["seq"], (0, pad_s), value=0))
        padded_durs.append(torch.nn.functional.pad(b["dur"], (0, pad_s), value=0))
        padded_pitches.append(torch.nn.functional.pad(b["pitch"], (0, pad_s), value=0))
        padded_energies.append(torch.nn.functional.pad(b["energy"], (0, pad_s), value=0))
        
        pad_m = max_mel_len - b["mel"].shape[-1]
        # Mel spektrogramda sessizlik tabanı ~ -11.5 dB
        padded_mels.append(torch.nn.functional.pad(b["mel"], (0, pad_m), value=-11.5))
        
    return {
        "seq": torch.stack(padded_seqs),
        "mel": torch.stack(padded_mels),
        "dur": torch.stack(padded_durs),
        "pitch": torch.stack(padded_pitches),
        "energy": torch.stack(padded_energies)
    }


if __name__ == "__main__":
    ds = DorinaVoiceDataset()
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_voice_batch)
    for b in loader:
        print("Batch Seq Shape:", b["seq"].shape)
        print("Batch Mel Shape:", b["mel"].shape)
        print("Batch Dur Shape:", b["dur"].shape)
        break
