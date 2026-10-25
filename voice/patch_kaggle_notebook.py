import json

with open('C:/Users/Talha/Documents/GitHub/dorina-ai/voice/verify_pipeline.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Dosya adı filtresini f.endswith('.wav') yap (Kaggle'a inen dosyalar train_00000.wav şeklinde)
fixed = text.replace('f.startswith("train_studio")', 'f.endswith(".wav")')
fixed = fixed.replace('data_dir: str = "voice/dataset_clean/wavs"', 'data_dir: str = "dataset/train"')
fixed = fixed.replace('steps: int = 50', 'steps: int = 300000')
fixed = fixed.replace('batch_size: int = 8', 'batch_size: int = 16')

# 2. Checkpoint bloğu ekle
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
fixed = fixed.replace('if step % 10 == 0 or step == 1:', checkpoint_logic + '\n        if step % 200 == 0 or step == 1:')

# 3. Çift main bloğunu tekilleştir
parts = fixed.split('if __name__ == "__main__":')
clean_body = parts[0] + 'if __name__ == "__main__":\n    train_pipeline(steps=300000, batch_size=16, data_dir="dataset/train")\n'

nb_path = 'C:/Users/Talha/Documents/GitHub/dorina-ai/voice/kaggle_deploy/dorina-vocoder-train.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][3]['source'] = ['# 2. Dogrulanmis HiFi-GAN Egitimi\n' + clean_body]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Kaggle notebook dosya filtresi cozuldu!')
