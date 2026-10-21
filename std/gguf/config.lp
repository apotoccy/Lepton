/* Model architecture configuration extracted from the GGUF KV block.

   The parser is deliberately independent of the kernel. It reads the
   metadata, with tensor-shape fallbacks for models that omit some keys,
   and the caller decides whether the running build supports the result.

   Keys follow the GGUF architecture convention:
     general.architecture
     <arch>.embedding_length
     <arch>.feed_forward_length
     <arch>.block_count
     <arch>.attention.head_count
     <arch>.attention.head_count_kv
     <arch>.rope.dimension_count
     <arch>.context_length
     <arch>.attention.layer_norm_rms_epsilon

   If a key is missing, the llama-prefixed key is tried, then tensor
   dimensions are used where possible. */

import "std/gguf/kv.lp";
import "std/gguf/tensor.lp";

struct ModelConfig {
    architecture: str;
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

fn config_key(arch: str, suffix: str) -> str {
    return "{arch}.{suffix}";
}

fn config_u32_arch(model: ptr[u8], kv_count: u64, arch: str,
                   suffix: str, fallback: i64) -> i64 {
    let key = config_key(arch, suffix);
    let v = config_u32(model, kv_count, key, -1);
    if (v >= 0) {
        return v;
    }
    let llama_key = config_key("llama", suffix);
    return config_u32(model, kv_count, llama_key, fallback);
}

fn config_f32_arch(model: ptr[u8], kv_count: u64, arch: str,
                   suffix: str, fallback: f32) -> f32 {
    let key = config_key(arch, suffix);
    let v = config_f32(model, kv_count, key, -1.0f);
    if (v > 0.0f) {
        return v;
    }
    let llama_key = config_key("llama", suffix);
    return config_f32(model, kv_count, llama_key, fallback);
}

fn tensor_blk_layer(model: ptr[u8], t: GGUFTensorInfo) -> i64 {
    if (t.name_len < 4 as u64) {
        return -1;
    }
    let off = t.name_offset;
    if (read_u8(model, off) != 98 as u8) { return -1; }      /* b */
    if (read_u8(model, off + 1) != 108 as u8) { return -1; }  /* l */
    if (read_u8(model, off + 2) != 107 as u8) { return -1; }  /* k */
    if (read_u8(model, off + 3) != 46 as u8) { return -1; }   /* . */
    var pos = off + 4;
    let end = off + (t.name_len as i64);
    var n: i64 = 0;
    var digits: i64 = 0;
    while (pos < end) {
        let c = read_u8(model, pos) as i64;
        if (c == 46) { break; }
        if (c < 48 || c > 57) { return -1; }
        n = n * 10 + (c - 48);
        digits = digits + 1;
        pos = pos + 1;
    }
    if (digits == 0) { return -1; }
    return n;
}

fn infer_layer_count(model: ptr[u8], tstart: i64, count: u64) -> i64 {
    var max_layer: i64 = -1;
    for i in 0..(count as i64) {
        let t = find_tensor_by_index(model, tstart, i as u64);
        let n = tensor_blk_layer(model, t);
        if (n > max_layer) { max_layer = n; }
    }
    return max_layer + 1;
}

fn load_model_config(model: ptr[u8], tstart: i64) -> ModelConfig {
    let kv_count = read_u64(model, 16);
    let tensor_count = read_u64(model, 8);

    var arch = "llama";
    let arch_off = find_kv(model, 24, kv_count, "general.architecture");
    if (arch_off >= 0) {
        let arch_len = read_u64(model, arch_off);
        arch = read_str(model, arch_off + 8, arch_len as i64);
    }

    let emb = tensor_by_name(model, tstart, tensor_count,
                             "token_embd.weight");

    var n_embd = config_u32_arch(model, kv_count, arch,
                                 "embedding_length", -1);
    if (n_embd < 0 && emb.name_len != 0 as u64) {
        n_embd = emb.dim0 as i64;
    }

    var n_ff = config_u32_arch(model, kv_count, arch,
                               "feed_forward_length", -1);
    if (n_ff < 0) {
        let gate = tensor_by_name(model, tstart, tensor_count,
                                  "blk.0.ffn_gate.weight");
        if (gate.name_len != 0 as u64) {
            n_ff = gate.dim0 as i64;
        } else {
            let up = tensor_by_name(model, tstart, tensor_count,
                                    "blk.0.ffn_up.weight");
            if (up.name_len != 0 as u64) {
                n_ff = up.dim0 as i64;
            }
        }
    }

    var n_layer = config_u32_arch(model, kv_count, arch,
                                  "block_count", -1);
    if (n_layer < 0) {
        n_layer = infer_layer_count(model, tstart, tensor_count);
    }

    var n_head = config_u32_arch(model, kv_count, arch,
                                 "attention.head_count", -1);
    var n_head_kv = config_u32_arch(model, kv_count, arch,
                                    "attention.head_count_kv", n_head);

    var head_dim = config_u32_arch(model, kv_count, arch,
                                   "rope.dimension_count", -1);
    if (head_dim < 0 && n_head > 0 && n_embd > 0) {
        head_dim = n_embd / n_head;
    }
    if (n_head < 0 && head_dim > 0 && n_embd > 0
            && n_embd % head_dim == 0) {
        n_head = n_embd / head_dim;
    }

    var max_seq = config_u32_arch(model, kv_count, arch,
                                  "context_length", 128);
    if (max_seq <= 0) {
        max_seq = 128;
    }
    /* The kernel KV/workspace budget is capped at 256 positions. */
    if (max_seq > 256) {
        max_seq = 256;
    }

    var rms_eps = config_f32_arch(
        model, kv_count, arch,
        "attention.layer_norm_rms_epsilon", -1.0f);
    if (rms_eps <= 0.0f) {
        rms_eps = config_f32_arch(
            model, kv_count, arch,
            "attention.layer_norm_epsilon", 1e-5f);
    }

    var vocab_size = 0 as i64;
    if (emb.name_len != 0 as u64) {
        vocab_size = emb.dim1 as i64;
    }

    return ModelConfig(
        architecture: arch,
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
