/* Model architecture configuration extracted from the GGUF KV block.

   The parser is deliberately independent of the kernel. It reads only
   the metadata and the token-embedding tensor shape; the caller decides
   whether the running build supports the resulting architecture.

   Keys follow the LLaMA GGUF conventions:
     llama.embedding_length
     llama.feed_forward_length
     llama.block_count
     llama.attention.head_count
     llama.attention.head_count_kv
     llama.rope.dimension_count
     llama.context_length
     llama.attention.layer_norm_rms_epsilon */

import "std/gguf/kv.lp";
import "std/gguf/tensor.lp";

struct ModelConfig {
    n_embd: i64;
    n_ff: i64;
    n_layer: i64;
    n_head: i64;
    n_head_kv: i64;
    head_dim: i64;
    max_seq: i64;
    vocab_size: i64;
    rms_eps: f32;
}

fn config_u32(model: ptr[u8], kv_count: u64, key: str,
              fallback: i64) -> i64 {
    let off = find_kv(model, 24, kv_count, key);
    if (off < 0) {
        return fallback;
    }
    return read_u32(model, off) as i64;
}

fn config_f32(model: ptr[u8], kv_count: u64, key: str,
              fallback: f32) -> f32 {
    let off = find_kv(model, 24, kv_count, key);
    if (off < 0) {
        return fallback;
    }
    return read_f32(model, off);
}

fn load_model_config(model: ptr[u8], tstart: i64) -> ModelConfig {
    let kv_count = read_u64(model, 16);

    let n_embd  = config_u32(model, kv_count,
                             "llama.embedding_length", -1);
    let n_ff    = config_u32(model, kv_count,
                             "llama.feed_forward_length", -1);
    let n_layer = config_u32(model, kv_count,
                             "llama.block_count", -1);
    let n_head  = config_u32(model, kv_count,
                             "llama.attention.head_count", -1);
    let n_head_kv = config_u32(model, kv_count,
                               "llama.attention.head_count_kv", n_head);

    var head_dim = config_u32(model, kv_count,
                              "llama.rope.dimension_count", -1);
    if (head_dim < 0 && n_head > 0 && n_embd > 0) {
        head_dim = n_embd / n_head;
    }

    var max_seq = config_u32(model, kv_count,
                             "llama.context_length", 128);
    if (max_seq <= 0) {
        max_seq = 128;
    }

    let rms_eps = config_f32(
        model, kv_count,
        "llama.attention.layer_norm_rms_epsilon", 1e-5f);

    let tensor_count = read_u64(model, 8);
    let emb = tensor_by_name(model, tstart, tensor_count,
                             "token_embd.weight");
    var vocab_size = 0 as i64;
    if (emb.name_len != 0 as u64) {
        vocab_size = emb.dim1 as i64;
    }

    return ModelConfig(
        n_embd: n_embd,
        n_ff: n_ff,
        n_layer: n_layer,
        n_head: n_head,
        n_head_kv: n_head_kv,
        head_dim: head_dim,
        max_seq: max_seq,
        vocab_size: vocab_size,
        rms_eps: rms_eps,
    );
}
