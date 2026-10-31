import json

with open('C:/Users/Talha/Documents/GitHub/dorina-ai/voice/verify_pipeline.py', 'r', encoding='utf-8') as f:
    full_code = f.read()

kaggle_train_code = full_code.replace('steps: int = 50', 'steps: int = 300000')
kaggle_train_code = kaggle_train_code.replace('batch_size: int = 8', 'batch_size: int = 16')
kaggle_train_code = kaggle_train_code.replace('voice/dataset_clean/wavs', 'dataset/train')

checkpoint_logic = """
        if step % 5000 == 0 or step == steps:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'step': step,
                'generator': generator.state_dict(),
                'mpd': mpd.state_dict(),
                'opt_g': opt_g.state_dict(),
                'opt_d': opt_d.state_dict()
            }, 'checkpoints/dorina_hifigan_active.pt')
            print(f'[CHECKPOINT] Adim {step} kaydedildi.')
"""

kaggle_train_code = kaggle_train_code.replace(
    'if step % 10 == 0 or step == 1:',
    checkpoint_logic + '\n        if step % 200 == 0 or step == 1:'
)

nb_path = 'C:/Users/Talha/Documents/GitHub/dorina-ai/voice/kaggle_deploy/dorina-vocoder-train.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'] = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['# Dorina HiFi-GAN V1 - Kaggle GPU Egitimi\n']
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': ['!pip install -q soundfile torchaudio pandas pyarrow\n']
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# 1. Veri Indirme (2.390 adet Turkce Kadin Sesi)\n',
            'import os, io, urllib.request\n',
            'import pandas as pd, soundfile as sf, torch, torchaudio, numpy as np\n',
            'os.makedirs("dataset/train", exist_ok=True)\n',
            'repos = [f"falan42/Turkish-Bentropi-tts-dataset-part-{i}" for i in range(2, 12)]\n',
            'total = 0\n',
            'for r in repos:\n',
            '    url = f"https://huggingface.co/datasets/{r}/resolve/main/data/train-00000-of-00001.parquet"\n',
            '    try:\n',
            '        df = pd.read_parquet(url)\n',
            '        for _, row in df.iterrows():\n',
            '            ab = row["audio"]["bytes"]\n',
            '            w, sr = sf.read(io.BytesIO(ab))\n',
            '            t_w = torch.tensor(w, dtype=torch.float32)\n',
            '            if t_w.dim() > 1: t_w = t_w.t()[:1]\n',
            '            else: t_w = t_w.unsqueeze(0)\n',
            '            if sr != 22050:\n',
            '                t_w = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)(t_w)\n',
            '            data_np = t_w.squeeze(0).numpy()\n',
            '            peak = np.max(np.abs(data_np)) + 1e-8\n',
            '            data_np = (data_np / peak) * 0.95\n',
            '            sf.write(f"dataset/train/train_{total:05d}.wav", data_np, 22050)\n',
            '            total += 1\n',
            '        print(f"[+] {r} indirildi. Toplam: {total}")\n',
            '    except Exception as e:\n',
            '        print(f"[!] {r} atlandi: {e}")\n',
            'print(f"[OK] Veri Hazir! Toplam {total} ses indirildi.")\n'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# 2. Dogrulanmis HiFi-GAN Egitimi\n' + kaggle_train_code + '\nif __name__ == "__main__":\n    train_pipeline(steps=300000, batch_size=16, data_dir="dataset/train")\n'
        ]
    }
]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Kaggle notebook kusursuz hale getirildi!')
