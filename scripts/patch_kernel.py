import os
import sys

def patch():
    with open('kaggle_kernel/train_v3_ddp.py', 'r', encoding='utf-8') as f:
        c = f.read()

    target1 = '    parser.add_argument("--save-interval", type=int, default=500)'
    replace1 = '    parser.add_argument("--save-interval", type=int, default=500)\n    parser.add_argument("--resume-step", type=int, default=0, help="Baslangic adimi")\n    parser.add_argument("--keep-last-n-ckpts", type=int, default=2, help="Kaggle disk korumasi icin tutulacak son n checkpoint")'

    assert target1 in c, "target1 not found"
    c = c.replace(target1, replace1, 1)

    target2 = '    for step in range(args.max_steps):'
    replace2 = '    saved_ckpts = []\n    for step in range(args.resume_step, args.max_steps):'
    assert target2 in c, "target2 not found"
    c = c.replace(target2, replace2, 1)

    target3 = '            print(f"  [CHECKPOINT] Kaydedildi: {ckpt_path}")'
    replace3 = '''            print(f"  [CHECKPOINT] Kaydedildi: {ckpt_path}")
            saved_ckpts.append(ckpt_path)
            if len(saved_ckpts) > args.keep_last_n_ckpts:
                to_del = saved_ckpts.pop(0)
                if os.path.exists(to_del):
                    try:
                        os.remove(to_del)
                        print(f"  [DISK KORUMA] Eski checkpoint silindi: {to_del}")
                    except Exception as e:
                        print(f"  [DISK KORUMA HATA]: {e}")'''
    assert target3 in c, "target3 not found"
    c = c.replace(target3, replace3, 1)

    with open('kaggle_kernel/train_v3_ddp.py', 'w', encoding='utf-8') as f:
        f.write(c)

    print("SUCCESS: train_v3_ddp.py patched!")

if __name__ == '__main__':
    patch()
