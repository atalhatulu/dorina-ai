import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gguf

def convert_dorina_to_gguf():
    print("[*] Loading Dorina-V4 SFT Checkpoint...")
    ckpt_path = "models/dorina_v4/dorina_v4_android_agent.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    weights = ckpt["model"]
    vocab_size = ckpt.get("vocab_size", 32832)

    # Architecture constants
    n_embd = 768
    n_layers = 16
    n_heads = 12
    n_kv_heads = 4
    head_dim = 64
    intermediate_size = 2560 # intermediate size
    max_seq_len = 1024

    output_path = "models/dorina_v4/dorina_v4_android_agent_f16.gguf"
    print(f"[*] Initializing GGUFWriter -> {output_path}")
    writer = gguf.GGUFWriter(output_path, "llama")

    # Add metadata
    writer.add_name("Dorina-V4-Android-Agent")
    writer.add_context_length(max_seq_len)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layers)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10000.0)
    writer.add_rope_dimension_count(head_dim)

    # Tokenizer
    print("[*] Loading Tokenizer...")
    with open("tokenizers/dorina_tokenizer_v4_sft.json", "r", encoding="utf-8") as f:
        tok_json = json.load(f)

    vocab = tok_json["model"]["vocab"]
    tokens = sorted(vocab.items(), key=lambda x: x[1])
    tokens_list = [t[0].encode("utf-8") for t in tokens]
    scores_list = [0.0] * len(tokens_list)
    tok_types = [gguf.TokenType.NORMAL] * len(tokens_list)

    # Pad tokens list to match vocab_size (32832)
    while len(tokens_list) < vocab_size:
        pad_idx = len(tokens_list)
        tokens_list.append(f"<extra_token_{pad_idx}>".encode("utf-8"))
        scores_list.append(0.0)
        tok_types.append(gguf.TokenType.USER_DEFINED)

    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens_list)
    writer.add_token_scores(scores_list)
    writer.add_token_types(tok_types)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)

    # Tensors mapping
    print("[*] Writing Model Tensors (FP16)...")
    
    # tok_embeddings
    tok_emb = weights["tok_embeddings.weight"].to(torch.float16).numpy()
    writer.add_tensor("token_embd.weight", tok_emb)
    # Output norm
    norm_f = weights["norm_f.weight"].to(torch.float32).numpy()
    writer.add_tensor("output_norm.weight", norm_f)
    # Output head (tied embeddings)
    writer.add_tensor("output.weight", tok_emb)

    for i in range(n_layers):
        # Attention Norm (norm1)
        writer.add_tensor(f"blk.{i}.attn_norm.weight", weights[f"layers.{i}.norm1.weight"].to(torch.float32).numpy())
        # Q, K, V proj
        writer.add_tensor(f"blk.{i}.attn_q.weight", weights[f"layers.{i}.attn.q_proj.weight"].to(torch.float16).numpy())
        writer.add_tensor(f"blk.{i}.attn_k.weight", weights[f"layers.{i}.attn.k_proj.weight"].to(torch.float16).numpy())
        writer.add_tensor(f"blk.{i}.attn_v.weight", weights[f"layers.{i}.attn.v_proj.weight"].to(torch.float16).numpy())
        writer.add_tensor(f"blk.{i}.attn_output.weight", weights[f"layers.{i}.attn.out_proj.weight"].to(torch.float16).numpy())

        # MLP Norm (norm2)
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", weights[f"layers.{i}.norm2.weight"].to(torch.float32).numpy())
        
        # In SwiGLU, w_in has shape [5120, 768] which is concat of gate and up (each 2560)
        w_in = weights[f"layers.{i}.mlp.w_in.weight"] # [5120, 768]
        w_gate = w_in[:intermediate_size, :].to(torch.float16).numpy()
        w_up = w_in[intermediate_size:, :].to(torch.float16).numpy()
        w_down = weights[f"layers.{i}.mlp.w_out.weight"].to(torch.float16).numpy() # [768, 2560]

        writer.add_tensor(f"blk.{i}.ffn_gate.weight", w_gate)
        writer.add_tensor(f"blk.{i}.ffn_up.weight", w_up)
        writer.add_tensor(f"blk.{i}.ffn_down.weight", w_down)

    print("[*] Flushing GGUF File...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[+] Successfully converted to GGUF: {output_path}")

if __name__ == "__main__":
    convert_dorina_to_gguf()
