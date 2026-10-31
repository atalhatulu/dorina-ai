import os
import sys
import json
import time
import re

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
from model_moe_v3 import DorinaMoEV3, MoEConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 50 Adet Standart Benchmark Görevini orijinal benchmark_suite'den import ediyoruz
from benchmark_suite import BENCHMARK_TASKS

def load_v3_for_benchmark(ckpt_path, tok_path):
    print(f"[Yükleniyor] V3 Benchmark Modeli: {ckpt_path} -> {device}")
    tokenizer = Tokenizer.from_file(tok_path)
    
    cfg = MoEConfig(
        vocab_size=32768,
        n_layer=12,
        d_model=512,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=1024,
        n_experts=32,
        top_k=2,
        expert_hidden=256,
        shared_hidden=256
    )
    
    model = DorinaMoEV3(cfg)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd)
    model.to(device)
    model.eval()
    return model, tokenizer

def v3_generate(model, tokenizer, prompt, max_new_tokens=90, temperature=0.5, top_k=25):
    input_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    gen_tokens = list(input_ids)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= 1024 else idx[:, -1024:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            gen_tokens.append(idx_next.item())
            if idx_next.item() == 3: # EOT
                break
                
    full_text = tokenizer.decode(gen_tokens)
    new_tokens = gen_tokens[len(input_ids):]
    completion = tokenizer.decode(new_tokens)
    return full_text, completion

def run_v3_benchmark(ckpt_path, out_json):
    tok_path = "tokenizers/dorina_tokenizer_v3_32k.json"
    model, tokenizer = load_v3_for_benchmark(ckpt_path, tok_path)
    
    category_scores = {}
    category_totals = {}
    total_score = 0.0
    task_results = []
    
    print("\n" + "=" * 70)
    print("DORINA V3 (180.6M) 50 SORULUK ALTIN BENCHMARK SINAVI")
    print("=" * 70)
    
    t0 = time.time()
    for i, task in enumerate(BENCHMARK_TASKS, 1):
        cat = task["category"]
        if cat not in category_scores:
            category_scores[cat] = 0.0
            category_totals[cat] = 0.0
        
        full_text, completion = v3_generate(model, tokenizer, task["prompt"])
        
        task_score = 0.0
        rubric_results = []
        for crit_name, pattern, max_points, desc in task["criteria"]:
            if re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE):
                task_score += max_points
                rubric_results.append({"crit": crit_name, "passed": True, "pts": max_points})
            else:
                rubric_results.append({"crit": crit_name, "passed": False, "pts": 0.0})
                
        category_scores[cat] += task_score
        category_totals[cat] += task["weight"]
        total_score += task_score
        
        print(f"[{i:02d}/50] ({cat[:14]:14s}) Skor: {task_score:.1f}/{task['weight']:.1f} | Soru: {task['id']}")
        task_results.append({
            "id": task["id"],
            "cat": cat,
            "score": task_score,
            "max": task["weight"],
            "prompt": task["prompt"],
            "completion": completion[:120].replace("\n", " ")
        })
        
    duration = time.time() - t0
    
    print("\n" + "=" * 70)
    print("🎯 DORİNA V3 (180.6M) SHARD 1 NİHAİ BENCHMARK KARNESİ")
    print("=" * 70)
    print(f"TOPLAM GENEL ZEKA PUANI: {total_score:.1f} / 100.0 (Süre: {duration:.1f} sn)")
    print("-" * 70)
    for cat in category_scores:
        pct = (category_scores[cat] / category_totals[cat]) * 100
        print(f"  * {cat:30s}: {category_scores[cat]:4.1f} / {category_totals[cat]:4.1f}  (%{pct:.1f})")
    print("=" * 70)
    
    report = {
        "model": "Dorina-V3-180M-Shard1",
        "total_score": total_score,
        "max_score": 100.0,
        "category_scores": category_scores,
        "duration_sec": duration,
        "tasks": task_results
    }
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Rapor kaydedildi: {out_json}")
    return total_score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="models/dorina_v3_180m/shard1/dorina_v3_shard1_final.pt")
    parser.add_argument("--out", default="data/dorina_v3_shard1_benchmark.json")
    args = parser.parse_args()
    run_v3_benchmark(args.ckpt, args.out)
