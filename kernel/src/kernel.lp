import "std/nn/matmul.lp";
import "std/gguf/kv.lp";
import "std/gguf/tensor.lp";
import "std/gguf/dequant.lp";

extern fn serial_init() -> void;
extern fn vga_init() -> void;
extern fn interrupts_init() -> void;
extern fn cpu_hlt() -> void;
extern fn ticks() -> u32;
extern fn kbd_read() -> u32;
extern fn mb_module_count() -> u32;
extern fn mb_module_start(idx: u32) -> u32;
extern fn mb_module_size(idx: u32) -> u32;
extern fn lp_argmax(logits: ptr[f32], n: i32) -> u32;
extern fn lp_sample(logits: ptr[f32], n: i32, temperature: f32, top_k: i32) -> u32;
extern fn lp_rand_seed(s: u32) -> void;
extern fn mb_module_end(idx: u32) -> u32;
extern fn lp_alloc_init(start: ptr[u8], size: i64) -> void;


fn starts_with(s: str, prefix: str) -> bool {
    var i = 0;
    while (prefix[i] != '\0') {
        if (s[i] != prefix[i]) {
            return false;
        }
        i = i + 1;
    }
    return true;
}

fn parse_int_at(text: str, start: i64) -> i64 {
    var i = start;
    while (text[i] == ' ') { i = i + 1; }
    var sign: i64 = 1;
    if (text[i] == '-') { sign = -1; i = i + 1; }
    var v: i64 = 0;
    while (text[i] != '\0') {
        let c = text[i] as i64;
        if (c < 48 || c > 57) { break; }
        v = v * 10 + (c - 48);
        i = i + 1;
    }
    return v * sign;
}

fn parse_int_end(text: str, start: i64) -> i64 {
    var i = start;
    while (text[i] == ' ') { i = i + 1; }
    if (text[i] == '-') { i = i + 1; }
    while (text[i] != '\0') {
        let c = text[i] as i64;
        if (c < 48 || c > 57) { break; }
        i = i + 1;
    }
    return i;
}

fn build_tok_offsets(model: ptr[u8], tok_table: ptr[u32]) -> i64 {
    let kc = read_u64(model, 16);
    let vo = find_kv(model, 24, kc, "tokenizer.ggml.tokens");
    if (vo < 0) { return -1; }
    let elem_type = read_u32(model, vo);
    if (elem_type != 8 as u32) { return -2; }
    let count = read_u64(model, vo + 4);
    var off = vo + 12;
    var i: i64 = 0;
    while (i < (count as i64)) {
        let len = read_u64(model, off);
        tok_table[i] = off as u32;
        off = off + 8 + (len as i64);
        i = i + 1;
    }
    return count as i64;
}

fn decode_tok(model: ptr[u8], tok_table: ptr[u32], tok_id: u32) -> str {
    let off = tok_table[tok_id as i64] as i64;
    let len = read_u64(model, off) as i64;
    return read_str(model, off + 8, len);
}

fn print_token(s: str) -> void {
    var len: i64 = 0;
    while (s[len] != '\0') { len = len + 1; }
    var i: i64 = 0;
    while (i < len) {
        if (i + 2 < len) {
            let b0 = (s[i] as u8) as i64;
            let b1 = (s[i+1] as u8) as i64;
            let b2 = (s[i+2] as u8) as i64;
            if (b0 == 0xE2 && b1 == 0x96 && b2 == 0x81) {
                print(" ");
                i = i + 3;
            } else {
                let c = s[i];
                print("{c}");
                i = i + 1;
            }
        } else {
            let c = s[i];
            print("{c}");
            i = i + 1;
        }
    }
}

fn name_match(buf: ptr[u8], off: i64, len: u64, target: str) -> bool {
    var i: i64 = 0;
    while (i < (len as i64)) {
        if (read_u8(buf, off + i) != target[i] as u8) {
            return false;
        }
        i = i + 1;
    }
    return target[i] == '\0';
}

fn find_tensor_index(model: ptr[u8], first_off: i64, count: u64, name: str) -> i64 {
    var walk: i64 = first_off;
    var k: i64 = 0;
    while (k < (count as i64)) {
        let nlen = read_u64(model, walk);
        let nstart = walk + 8;
        if (name_match(model, nstart, nlen, name)) {
            return k;
        }
        let ndims = read_u32(model, nstart + (nlen as i64));
        walk = nstart + (nlen as i64) + 4 + 8 * (ndims as i64) + 4 + 8;
        k = k + 1;
    }
    return -1;
}

fn tensor_by_name(model: ptr[u8], first_off: i64, count: u64, name: str) -> GGUFTensorInfo {
    let idx = find_tensor_index(model, first_off, count, name);
    if (idx < 0) {
        return GGUFTensorInfo(
            name_offset: 0, name_len: 0, n_dims: 0,
            dim0: 0, dim1: 0, dim2: 0, dim3: 0,
            ggml_type: 0, data_offset: 0,
        );
    }
    return find_tensor_by_index(model, first_off, idx as u64);
}

fn model_data_start(model: ptr[u8]) -> i64 {
    let tc = read_u64(model, 8);
    let kc = read_u64(model, 16);
    let walk_end = skip_kv_block(model, 24, kc);
    var mds: i64 = walk_end;
    for i in 0..(tc as i64) {
        mds = mds + tensor_info_size(model, mds);
    }
    let rem = mds % 32;
    if (rem != 0) {
        mds = mds + (32 - rem);
    }
    return mds;
}

fn dequant_row_q4_0(model: ptr[u8], data_start: i64,
                     t: GGUFTensorInfo, row: i64, out: ptr[f32]) -> void {
    let dim = t.dim0 as i64;
    let blocks = dim / 32;
    let row_bytes = blocks * 18;
    let src = (model + data_start + (t.data_offset as i64) + row * row_bytes) as ptr[u8];
    dequant_q4_0_tensor(src, out, blocks);
}

fn dequant_tensor_q4_0(model: ptr[u8], data_start: i64,
                        t: GGUFTensorInfo, out: ptr[f32]) -> void {
    let n_total = (t.dim0 * t.dim1) as i64;
    let blocks = n_total / 32;
    let src = (model + data_start + (t.data_offset as i64)) as ptr[u8];
    dequant_q4_0_tensor(src, out, blocks);
}

fn copy_f32_tensor(model: ptr[u8], data_start: i64,
                    t: GGUFTensorInfo, out: ptr[f32]) -> void {
    let n_total = (t.dim0 * t.dim1) as i64;
    let src = (model + data_start + (t.data_offset as i64)) as ptr[f32];
    for i in 0..n_total {
        out[i] = src[i];
    }
}

fn rmsnorm_n(x: ptr[f32], w: ptr[f32], out: ptr[f32], n: i64) -> void {
    var ss: f32 = 0.0f;
    for i in 0..n {
        ss = ss + x[i] * x[i];
    }
    let ms = ss / (n as f32);
    let inv = rsqrt(ms + 1e-5f);
    for i in 0..n {
        out[i] = x[i] * inv * w[i];
    }
}

fn matvec_rows(W: ptr[f32], x: ptr[f32], y: ptr[f32],
                rows: i64, cols: i64) -> void {
    for i in 0..rows {
        var mv_acc: f32 = 0.0f;
        for j in 0..cols {
            mv_acc = mv_acc + W[i * cols + j] * x[j];
        }
        y[i] = mv_acc;
    }
}

fn silu_f32(x: f32) -> f32 {
    return x / (1.0f + exp(0.0f - x));
}

fn build_rope_freqs(freqs: ptr[f64], head_dim: i64) -> void {
    /* freqs[i] = 10000^(-2i/head_dim) for i in 0..head_dim/2 */
    let ln_10000: f64 = 9.210340371976184;
    let half = head_dim / 2;
    for i in 0..half {
        let exponent: f64 = (0.0 - 2.0) * (i as f64) / (head_dim as f64) * ln_10000;
        freqs[i] = exp(exponent);
    }
}

fn rope_apply(x: ptr[f32], pos: i64, freqs: ptr[f64],
              n_head: i64, head_dim: i64) -> void {
    /* Interleaved convention: pairs (2i, 2i+1) within each head, rotated by pos*freqs[i]. */
    let half = head_dim / 2;
    for h in 0..n_head {
        for i in 0..half {
            let angle: f64 = (pos as f64) * freqs[i];
            let c = cos(angle);
            let s = sin(angle);
            let base = h * head_dim + i * 2;
            let a = x[base] as f64;
            let b = x[base + 1] as f64;
            x[base]     = (a * c - b * s) as f32;
            x[base + 1] = (a * s + b * c) as f32;
        }
    }
}

fn softmax_inplace(x: ptr[f32], n: i64) -> void {
    var maxv: f32 = x[0];
    for i in 1..n {
        if (x[i] > maxv) { maxv = x[i]; }
    }
    var total: f32 = 0.0f;
    for i in 0..n {
        let e = exp(x[i] - maxv);
        x[i] = e;
        total = total + e;
    }
    let inv = 1.0f / total;
    for i in 0..n {
        x[i] = x[i] * inv;
    }
}

fn forward_token(model: ptr[u8], token: i64, n_layer: i64,
                 wbuf: ptr[f32], act: ptr[f32],
                 out_token: ptr[u32],
                 trace: bool) -> void {
    let data_start = model_data_start(model);
    let tc = read_u64(model, 8);
    let kc = read_u64(model, 16);
    let tstart = skip_kv_block(model, 24, kc);

    let n_embd: i64 = 288;
    let n_ff: i64   = 768;

    /* Act buffer layout — all f32, offsets in floats.
       [0..288]        hidden
       [288..576]      normed
       [576..864]      v_vec
       [864..1152]     attn_out
       [1152..1920]    ffn_gate (768)
       [1920..2688]    ffn_up   (768)
       [2688..2976]    ffn_down (288)
       [2976..3264]    logits_row (288) */
    let hidden   = act;
    let normed   = act + 288;
    let v_vec    = act + 576;
    let attn_out = act + 864;
    let ffn_gate = act + 1152;
    let ffn_up   = act + 1920;
    let ffn_down = act + 2688;
    let logit_r  = act + 2976;

    /* 1. Embedding */
    let emb_t = tensor_by_name(model, tstart, tc, "token_embd.weight");
    dequant_row_q4_0(model, data_start, emb_t, token, hidden);
    if (trace) {
        println("  emb: {hidden[0]} {hidden[1]} {hidden[2]}");
    }

    /* 2. Layers */
    for layer in 0..n_layer {
        /* 2a. RMSNorm 1 */
        let an_name = "blk.{layer}.attn_norm.weight";
        let an_t = tensor_by_name(model, tstart, tc, an_name);
        copy_f32_tensor(model, data_start, an_t, wbuf);
        rmsnorm_n(hidden, wbuf, normed, n_embd);

        /* 2b. V = attn_v @ normed */
        let av_name = "blk.{layer}.attn_v.weight";
        let av_t = tensor_by_name(model, tstart, tc, av_name);
        dequant_tensor_q4_0(model, data_start, av_t, wbuf + 288);
        matvec_rows(wbuf + 288, normed, v_vec, n_embd, n_embd);

        /* 2c. attn_out = attn_output @ V */
        let ao_name = "blk.{layer}.attn_output.weight";
        let ao_t = tensor_by_name(model, tstart, tc, ao_name);
        dequant_tensor_q4_0(model, data_start, ao_t, wbuf + 288);
        matvec_rows(wbuf + 288, v_vec, attn_out, n_embd, n_embd);

        /* 2d. Residual */
        for i in 0..n_embd {
            hidden[i] = hidden[i] + attn_out[i];
        }
        if (trace && layer == 0) {
            println("  L0 attn: {hidden[0]} {hidden[1]} {hidden[2]}");
        }

        /* 2e. RMSNorm 2 */
        let fn_name = "blk.{layer}.ffn_norm.weight";
        let fn_t = tensor_by_name(model, tstart, tc, fn_name);
        copy_f32_tensor(model, data_start, fn_t, wbuf);
        rmsnorm_n(hidden, wbuf, normed, n_embd);

        /* 2f. gate = ffn_gate @ normed */
        let fg_name = "blk.{layer}.ffn_gate.weight";
        let fg_t = tensor_by_name(model, tstart, tc, fg_name);
        dequant_tensor_q4_0(model, data_start, fg_t, wbuf);
        matvec_rows(wbuf, normed, ffn_gate, n_ff, n_embd);

        /* 2g. up = ffn_up @ normed */
        let fu_name = "blk.{layer}.ffn_up.weight";
        let fu_t = tensor_by_name(model, tstart, tc, fu_name);
        dequant_tensor_q4_0(model, data_start, fu_t, wbuf);
        matvec_rows(wbuf, normed, ffn_up, n_ff, n_embd);

        /* 2h. silu_gate = silu(gate) * up, stored back into ffn_up */
        for i in 0..n_ff {
            ffn_up[i] = silu_f32(ffn_gate[i]) * ffn_up[i];
        }

        /* 2i. ffn_down = ffn_down_w @ silu_gate */
        let fd_name = "blk.{layer}.ffn_down.weight";
        let fd_t = tensor_by_name(model, tstart, tc, fd_name);
        dequant_tensor_q4_0(model, data_start, fd_t, wbuf);
        matvec_rows(wbuf, ffn_up, ffn_down, n_embd, n_ff);

        /* 2j. Residual */
        for i in 0..n_embd {
            hidden[i] = hidden[i] + ffn_down[i];
        }
        if (trace && layer == 0) {
            println("  L0 ffn:  {hidden[0]} {hidden[1]} {hidden[2]}");
        }
    }
    if (trace) {
        println("  final:  {hidden[0]} {hidden[1]} {hidden[2]}");
    }

    /* 3. Final RMSNorm */
    let onf_t = tensor_by_name(model, tstart, tc, "output_norm.weight");
    copy_f32_tensor(model, data_start, onf_t, wbuf);
    rmsnorm_n(hidden, wbuf, normed, n_embd);

    /* 4. Output projection, per-vocab-row dequant, argmax.
       output.weight can be type=2 (Q4_0) or type=8 (Q8_0); handle both. */
    let ow_t = tensor_by_name(model, tstart, tc, "output.weight");
    let v_size = ow_t.dim1 as i64;
    var best_l: f32 = -3.0e38f;
    var best_t: u32 = 0 as u32;
    for vv in 0..v_size {
        if (ow_t.ggml_type == 2 as u32) {
            dequant_row_q4_0(model, data_start, ow_t, vv, logit_r);
        } else {
            /* Q8_0: 34 bytes per 32 elements. */
            let blocks = n_embd / 32;
            let row_bytes = blocks * 34;
            let src = (model + data_start + (ow_t.data_offset as i64) + vv * row_bytes) as ptr[u8];
            dequant_q8_0_tensor(src, logit_r, blocks);
        }
        var lp_dot: f32 = 0.0f;
        for j in 0..n_embd {
            lp_dot = lp_dot + logit_r[j] * normed[j];
        }
        if (lp_dot > best_l) {
            best_l = lp_dot;
            best_t = vv as u32;
        }
    }
    out_token[0] = best_t;
}

fn forward_token_at_pos(model: ptr[u8], token: i64, pos: i64,
                        n_layer: i64, n_head: i64, head_dim: i64,
                        wbuf: ptr[f32], act: ptr[f32],
                        kv_k: ptr[f32], kv_v: ptr[f32],
                        freqs: ptr[f64],
                        logits: ptr[f32]) -> void {
    let data_start = model_data_start(model);
    let tc = read_u64(model, 8);
    let kc = read_u64(model, 16);
    let tstart = skip_kv_block(model, 24, kc);

    let n_embd: i64 = 288;
    let n_ff:   i64 = 768;
    let max_seq: i64 = 256;

    let hidden    = act;
    let normed    = act + 288;
    let q_buf     = act + 576;
    let k_buf     = act + 864;
    let v_buf     = act + 1152;
    let attended  = act + 1440;
    let attn_out  = act + 1728;
    let gate      = act + 2016;
    let up        = act + 2784;
    let silu_gate = act + 3552;
    let down      = act + 4320;
    let final_n   = act + 4608;
    let logit_r   = act + 4896;
    let scores    = act + 5184;

    /* 1. Embedding */
    let emb_t = tensor_by_name(model, tstart, tc, "token_embd.weight");
    dequant_row_q4_0(model, data_start, emb_t, token, hidden);

    /* 2. Layers */
    for layer in 0..n_layer {
        /* RMSNorm 1 */
        let an_t = tensor_by_name(model, tstart, tc, "blk.{layer}.attn_norm.weight");
        copy_f32_tensor(model, data_start, an_t, wbuf);
        rmsnorm_n(hidden, wbuf, normed, n_embd);

        /* Q, K, V */
        let aq_t = tensor_by_name(model, tstart, tc, "blk.{layer}.attn_q.weight");
        dequant_tensor_q4_0(model, data_start, aq_t, wbuf);
        matvec_rows(wbuf, normed, q_buf, n_embd, n_embd);

        let ak_t = tensor_by_name(model, tstart, tc, "blk.{layer}.attn_k.weight");
        dequant_tensor_q4_0(model, data_start, ak_t, wbuf);
        matvec_rows(wbuf, normed, k_buf, n_embd, n_embd);

        let av_t = tensor_by_name(model, tstart, tc, "blk.{layer}.attn_v.weight");
        dequant_tensor_q4_0(model, data_start, av_t, wbuf);
        matvec_rows(wbuf, normed, v_buf, n_embd, n_embd);

        /* RoPE (identity at pos=0) */
        rope_apply(q_buf, pos, freqs, n_head, head_dim);
        rope_apply(k_buf, pos, freqs, n_head, head_dim);

        /* Cache K, V at [layer][pos] */
        let cache_base = (layer * max_seq + pos) * n_embd;
        for i in 0..n_embd {
            kv_k[cache_base + i] = k_buf[i];
            kv_v[cache_base + i] = v_buf[i];
        }

        /* Attention over positions 0..pos */
        let scale: f32 = 1.0f / sqrt(head_dim as f32);
        for h in 0..n_head {
            let q_off = h * head_dim;
            for p in 0..(pos + 1) {
                let kbase = (layer * max_seq + p) * n_embd + q_off;
                var acc_s: f32 = 0.0f;
                for d in 0..head_dim {
                    acc_s = acc_s + q_buf[q_off + d] * kv_k[kbase + d];
                }
                scores[p] = acc_s * scale;
            }
            softmax_inplace(scores, pos + 1);
            for d in 0..head_dim {
                var acc_v: f32 = 0.0f;
                for p in 0..(pos + 1) {
                    let vbase = (layer * max_seq + p) * n_embd + q_off;
                    acc_v = acc_v + scores[p] * kv_v[vbase + d];
                }
                attended[q_off + d] = acc_v;
            }
        }

        /* attn_out = Wo @ attended */
        let ao_t = tensor_by_name(model, tstart, tc, "blk.{layer}.attn_output.weight");
        dequant_tensor_q4_0(model, data_start, ao_t, wbuf);
        matvec_rows(wbuf, attended, attn_out, n_embd, n_embd);

        for i in 0..n_embd {
            hidden[i] = hidden[i] + attn_out[i];
        }

        /* RMSNorm 2 */
        let fn_t = tensor_by_name(model, tstart, tc, "blk.{layer}.ffn_norm.weight");
        copy_f32_tensor(model, data_start, fn_t, wbuf);
        rmsnorm_n(hidden, wbuf, normed, n_embd);

        /* FFN */
        let fg_t = tensor_by_name(model, tstart, tc, "blk.{layer}.ffn_gate.weight");
        dequant_tensor_q4_0(model, data_start, fg_t, wbuf);
        matvec_rows(wbuf, normed, gate, n_ff, n_embd);

        let fu_t = tensor_by_name(model, tstart, tc, "blk.{layer}.ffn_up.weight");
        dequant_tensor_q4_0(model, data_start, fu_t, wbuf);
        matvec_rows(wbuf, normed, up, n_ff, n_embd);

        for i in 0..n_ff {
            silu_gate[i] = silu_f32(gate[i]) * up[i];
        }

        let fd_t = tensor_by_name(model, tstart, tc, "blk.{layer}.ffn_down.weight");
        dequant_tensor_q4_0(model, data_start, fd_t, wbuf);
        matvec_rows(wbuf, silu_gate, down, n_embd, n_ff);

        for i in 0..n_embd {
            hidden[i] = hidden[i] + down[i];
        }
    }

    /* Final RMSNorm */
    let onf_t = tensor_by_name(model, tstart, tc, "output_norm.weight");
    copy_f32_tensor(model, data_start, onf_t, wbuf);
    rmsnorm_n(hidden, wbuf, final_n, n_embd);

    /* Output projection: fill logits[vocab] with raw scores */
    let ow_t = tensor_by_name(model, tstart, tc, "output.weight");
    let v_size = ow_t.dim1 as i64;
    for vv in 0..v_size {
        if (ow_t.ggml_type == 2 as u32) {
            dequant_row_q4_0(model, data_start, ow_t, vv, logit_r);
        } else {
            let blocks = n_embd / 32;
            let row_bytes = blocks * 34;
            let src = (model + data_start + (ow_t.data_offset as i64) + vv * row_bytes) as ptr[u8];
            dequant_q8_0_tensor(src, logit_r, blocks);
        }
        var lp_dot: f32 = 0.0f;
        for j in 0..n_embd {
            lp_dot = lp_dot + logit_r[j] * final_n[j];
        }
        logits[vv] = lp_dot;
    }
}

fn parse_float_at(text: str, start: i64) -> f32 {
    var i = start;
    while (text[i] == ' ') { i = i + 1; }
    var sign: f32 = 1.0f;
    if (text[i] == '-') { sign = -1.0f; i = i + 1; }
    var ip: f32 = 0.0f;
    while (text[i] != '\0') {
        let c = text[i] as i64;
        if (c < 48 || c > 57) { break; }
        ip = ip * 10.0f + ((c - 48) as f32);
        i = i + 1;
    }
    var fp: f32 = 0.0f;
    var div: f32 = 1.0f;
    if (text[i] == '.') {
        i = i + 1;
        while (text[i] != '\0') {
            let c = text[i] as i64;
            if (c < 48 || c > 57) { break; }
            fp = fp * 10.0f + ((c - 48) as f32);
            div = div * 10.0f;
            i = i + 1;
        }
    }
    return sign * (ip + fp / div);
}

fn parse_float_end(text: str, start: i64) -> i64 {
    var i = start;
    while (text[i] == ' ') { i = i + 1; }
    if (text[i] == '-') { i = i + 1; }
    while (text[i] != '\0') {
        let c = text[i] as i64;
        if ((c >= 48 && c <= 57) || c == 46) {
            i = i + 1;
        } else {
            break;
        }
    }
    return i;
}

fn build_merge_offsets(model: ptr[u8], merge_table: ptr[u32]) -> i64 {
    let kc = read_u64(model, 16);
    let vo = find_kv(model, 24, kc, "tokenizer.ggml.merges");
    if (vo < 0) { return -1; }
    let et = read_u32(model, vo);
    if (et != 8 as u32) { return -2; }
    let count = read_u64(model, vo + 4);
    var off = vo + 12;
    var i: i64 = 0;
    while (i < (count as i64)) {
        let len = read_u64(model, off);
        merge_table[i] = off as u32;
        off = off + 8 + (len as i64);
        i = i + 1;
    }
    return count as i64;
}

fn piece_eq(model: ptr[u8], a_off: i64, a_len: i64,
            b_off: i64, b_len: i64) -> bool {
    if (a_len != b_len) { return false; }
    var i: i64 = 0;
    while (i < a_len) {
        if (read_u8(model, a_off + i) != read_u8(model, b_off + i)) {
            return false;
        }
        i = i + 1;
    }
    return true;
}

fn bpe_encode_word(model: ptr[u8],
                   tok_table: ptr[u32], tok_count: i64,
                   merge_table: ptr[u32], merge_count: i64,
                   word_buf: ptr[u8], word_len: i64,
                   pieces_buf: ptr[u8],
                   pieces_len: ptr[u32],
                   out_ids: ptr[u32]) -> i64 {
    /* If merges are absent (this model has no tokenizer.ggml.merges),
       fall back to greedy longest-match against the vocab. For BPE-
       trained vocabularies this produces the same tokenisation as
       applying all merges in rank order, because a piece only exists
       in the vocab if it appeared as a merge result. */
    if (merge_count <= 0) {
        var n: i64 = 0;
        var pos: i64 = 0;
        while (pos < word_len) {
            var best_len: i64 = 0;
            var best_id: i64 = -1;

            var cand_len: i64 = word_len - pos;
            if (cand_len > 32) { cand_len = 32; }
            while (cand_len > 0) {
                var tok: i64 = 0;
                var found: bool = false;
                while (tok < tok_count) {
                    let toff = tok_table[tok] as i64;
                    let tlen = read_u64(model, toff) as i64;
                    if (tlen == cand_len) {
                        var k: i64 = 0;
                        var match: bool = true;
                        while (k < cand_len) {
                            if (read_u8(model, toff + 8 + k) != word_buf[pos + k]) {
                                match = false;
                                break;
                            }
                            k = k + 1;
                        }
                        if (match) {
                            best_len = cand_len;
                            best_id = tok;
                            found = true;
                            break;
                        }
                    }
                    tok = tok + 1;
                }
                if (found) { break; }
                cand_len = cand_len - 1;
            }

            if (best_id < 0) {
                out_ids[n] = 0 as u32;
                n = n + 1;
                pos = pos + 1;
            } else {
                out_ids[n] = best_id as u32;
                n = n + 1;
                pos = pos + best_len;
            }
        }
        return n;
    }

    /* Merge-based BPE path. Only reached if a merges list is loaded. */
    var m_n: i64 = 0;
    var m_i: i64 = 0;
    while (m_i < word_len) {
        pieces_buf[m_i * 64] = word_buf[m_i];
        pieces_len[m_i] = 1 as u32;
        m_n = m_n + 1;
        m_i = m_i + 1;
    }

    var progress: bool = true;
    while (progress) {
        progress = false;
        var m: i64 = 0;
        while (m < merge_count) {
            let moff = merge_table[m] as i64;
            let mlen = read_u64(model, moff) as i64;
            let mstr = moff + 8;

            var space_at: i64 = -1;
            var kk: i64 = 0;
            while (kk < mlen) {
                if (read_u8(model, mstr + kk) == 32) {
                    space_at = kk;
                    break;
                }
                kk = kk + 1;
            }
            if (space_at < 0) { m = m + 1; continue; }

            let x_off = mstr;
            let x_len = space_at;
            let y_off = mstr + space_at + 1;
            let y_len = mlen - space_at - 1;

            var p: i64 = 0;
            var found_p: i64 = -1;
            while (p < m_n - 1) {
                let p1_len = pieces_len[p] as i64;
                let p2_len = pieces_len[p+1] as i64;
                if (piece_eq(model, x_off, x_len, p * 64, p1_len)
                        && piece_eq(model, y_off, y_len, (p+1) * 64, p2_len)) {
                    found_p = p;
                    break;
                }
                p = p + 1;
            }
            if (found_p < 0) { m = m + 1; continue; }

            let dst_base = found_p * 64;
            let src_base = (found_p + 1) * 64;
            let dst_off = pieces_len[found_p] as i64;
            let src_len = pieces_len[found_p + 1] as i64;
            var q: i64 = 0;
            while (q < src_len) {
                pieces_buf[dst_base + dst_off + q] = pieces_buf[src_base + q];
                q = q + 1;
            }
            pieces_len[found_p] = (dst_off + src_len) as u32;

            var r: i64 = found_p + 1;
            while (r < m_n - 1) {
                var q2: i64 = 0;
                while (q2 < 64) {
                    pieces_buf[r * 64 + q2] = pieces_buf[(r+1) * 64 + q2];
                    q2 = q2 + 1;
                }
                pieces_len[r] = pieces_len[r + 1];
                r = r + 1;
            }
            m_n = m_n - 1;
            progress = true;
            break;
        }
    }

    var t: i64 = 0;
    while (t < m_n) {
        let piece_len = pieces_len[t] as i64;
        var id: i64 = -1;
        var tok: i64 = 0;
        while (tok < tok_count) {
            let toff = tok_table[tok] as i64;
            let tlen = read_u64(model, toff) as i64;
            if (tlen == piece_len) {
                let tstr = toff + 8;
                if (piece_eq(model, tstr, tlen, t * 64, piece_len)) {
                    id = tok;
                    break;
                }
            }
            tok = tok + 1;
        }
        if (id < 0) {
            id = 0;
        }
        out_ids[t] = id as u32;
        t = t + 1;
    }
    return m_n;
}

fn encode_prompt(model: ptr[u8],
                 tok_table: ptr[u32], tok_count: i64,
                 merge_table: ptr[u32], merge_count: i64,
                 text_buf: ptr[u8], text_start: i64, text_end: i64,
                 out_ids: ptr[u32]) -> i64 {
    let word_buf = alloc([u8; 260]) as ptr[u8];
    let pieces_buf = alloc([u8; 8192]) as ptr[u8];
    let pieces_len = alloc([u32; 128]) as ptr[u32];
    let temp_ids = alloc([u32; 128]) as ptr[u32];

    var total: i64 = 0;
    var word_len: i64 = 0;
    var i: i64 = text_start;

    while (i <= text_end) {
        let c = read_u8(text_buf, i);
        if (i == text_end || c == 32 || c == 10 || c == 13) {
            if (word_len > 0) {
                /* Prepend ▁ = E2 96 81 */
                let full_buf = alloc([u8; 264]) as ptr[u8];
                full_buf[0] = 0xE2 as u8;
                full_buf[1] = 0x96 as u8;
                full_buf[2] = 0x81 as u8;
                var w: i64 = 0;
                while (w < word_len) {
                    full_buf[3 + w] = word_buf[w];
                    w = w + 1;
                }
                let np = bpe_encode_word(model, tok_table, tok_count,
                                          merge_table, merge_count,
                                          full_buf, word_len + 3,
                                          pieces_buf, pieces_len, temp_ids);
                var j: i64 = 0;
                while (j < np) {
                    out_ids[total + j] = temp_ids[j];
                    j = j + 1;
                }
                total = total + np;
                word_len = 0;
            }
            i = i + 1;
        } else {
            word_buf[word_len] = c;
            word_len = word_len + 1;
            i = i + 1;
        }
    }
    return total;
}

fn parse_quoted(text: str, start: i64, out_buf: ptr[u8]) -> i64 {
    var i = start;
    while (text[i] == ' ') { i = i + 1; }
    if (text[i] != '"') { return -1; }
    i = i + 1;
    var n: i64 = 0;
    while (text[i] != '"' && text[i] != '\0') {
        out_buf[n] = text[i] as u8;
        n = n + 1;
        i = i + 1;
    }
    if (text[i] == '"') {
        out_buf[n] = 0 as u8;
        return i + 1;
    }
    return -1;
}

fn main() {
    serial_init();
    vga_init();

    /* Place the heap 1 MB AFTER the loaded model ends */
    var heap_addr: u32 = 16777216 as u32; /* 16MB fallback */
    if (mb_module_count() > 0 as u32) {
        heap_addr = mb_module_end(0 as u32) + 1048576 as u32;
    }
    lp_alloc_init(heap_addr as ptr[u8], 67108864); /* 64 MB of safe RAM */

    println("Lepton OS kernel");
    println("");

    /* ---- 1. FPU / SSE verification ---- */
    println("[1] FPU / SSE");
    let s = sqrt(2.0);
    println("    sqrt(2) = {s}");
    let e = exp(1.0);
    println("    exp(1)  = {e}");
    let t = tanh(1.0);
    println("    tanh(1) = {t}");
    println("");

    /* ---- 2. Fixed-point arithmetic ---- */
    println("[2] Fixed-point (q15)");
    let a = 0.5 as q15;
    let b = 0.25 as q15;
    let c = a + b;
    println("    0.5 + 0.25 = {c}");
    let d = a * b;
    println("    0.5 * 0.25 = {d}");
    println("");

    /* ---- 3. Heap allocation + matvec ---- */
    println("[3] Heap + matvec");
    let m = alloc([f32; 16]);
    for i in 0..16 {
        m[i] = 0.0f;
    }
    m[0]  = 2.0f;
    m[5]  = 3.0f;
    m[10] = 4.0f;
    m[15] = 5.0f;
    let v = [1.0f, 1.0f, 1.0f, 1.0f];
    let r = matvec4(m, v);
    println("    diag[2,3,4,5] * ones = {r[0]} {r[1]} {r[2]} {r[3]}");
    println("");

    /* ---- 4. Compute pi via Leibniz series ---- */
    println("[4] Computing pi (Leibniz, 10000 terms)");
    var pi = 0.0;
    for k in 0..10000 {
        let kf = k as f64;
        let term = 1.0 / (2.0 * kf + 1.0);
        if (k % 2 == 0) {
            pi = pi + term;
        } else {
            pi = pi - term;
        }
    }
    pi = pi * 4.0;
    println("    pi = {pi}");
    println("");

    /* ---- 5. ASCII circle drawn with real FPU comparisons ---- */
    println("[5] ASCII circle (FPU distance checks)");
    for row in 0..15 {
        for col in 0..40 {
            let x = (col as f64 - 20.0) / 10.0;
            let y = (row as f64 - 7.5) / 6.0;
            let dist = sqrt(x * x + y * y);
            if (dist < 1.0) {
                print("#");
            } else if (dist < 1.1) {
                print(".");
            } else {
                print(" ");
            }
        }
        println("");
    }
    println("");
    println("done.");

    println("[5a] FPU check: sqrt(2) = {sqrt(2.0)}");

    /* ---- 6. Boot banner "LEPTON OS" from a 5x7 bitmap font ---- */
    println("[6] Boot banner");
    let font = alloc([u8; 63]);

    /* L */
    font[0]  = 0x10 as u8;  font[1]  = 0x10 as u8;  font[2]  = 0x10 as u8;
    font[3]  = 0x10 as u8;  font[4]  = 0x10 as u8;  font[5]  = 0x10 as u8;
    font[6]  = 0x1F as u8;

    /* E */
    font[7]  = 0x1F as u8;  font[8]  = 0x10 as u8;  font[9]  = 0x10 as u8;
    font[10] = 0x1E as u8;  font[11] = 0x10 as u8;  font[12] = 0x10 as u8;
    font[13] = 0x1F as u8;

    /* P */
    font[14] = 0x1E as u8;  font[15] = 0x11 as u8;  font[16] = 0x11 as u8;
    font[17] = 0x1E as u8;  font[18] = 0x10 as u8;  font[19] = 0x10 as u8;
    font[20] = 0x10 as u8;

    /* T */
    font[21] = 0x1F as u8;  font[22] = 0x04 as u8;  font[23] = 0x04 as u8;
    font[24] = 0x04 as u8;  font[25] = 0x04 as u8;  font[26] = 0x04 as u8;
    font[27] = 0x04 as u8;

    /* O */
    font[28] = 0x0E as u8;  font[29] = 0x11 as u8;  font[30] = 0x11 as u8;
    font[31] = 0x11 as u8;  font[32] = 0x11 as u8;  font[33] = 0x11 as u8;
    font[34] = 0x0E as u8;

    /* N */
    font[35] = 0x11 as u8;  font[36] = 0x19 as u8;  font[37] = 0x15 as u8;
    font[38] = 0x13 as u8;  font[39] = 0x11 as u8;  font[40] = 0x11 as u8;
    font[41] = 0x11 as u8;

    /* space (all zeros) */
    for i in 42..49 { font[i] = 0 as u8; }

    /* O */
    font[49] = 0x0E as u8;  font[50] = 0x11 as u8;  font[51] = 0x11 as u8;
    font[52] = 0x11 as u8;  font[53] = 0x11 as u8;  font[54] = 0x11 as u8;
    font[55] = 0x0E as u8;

    /* S */
    font[56] = 0x0F as u8;  font[57] = 0x10 as u8;  font[58] = 0x10 as u8;
    font[59] = 0x0E as u8;  font[60] = 0x01 as u8;  font[61] = 0x01 as u8;
    font[62] = 0x1E as u8;

    /* Render: for each row, for each of the 9 glyphs, for each of 5 columns. */
    for row in 0..7 {
        for letter in 0..9 {
            let bits = font[letter * 7 + row] as i64;
            var mask: i64 = 16;
            for col in 0..5 {
                if ((bits & mask) != 0) {
                    print("#");
                } else {
                    print(" ");
                }
                mask = mask >> 1;
            }
            print(" ");
        }
        println("");
    }
    println("");

    println("");

    /* ---- 7. Interrupts + interactive shell ---- */
    let scratch_bytes = alloc([u8; 4194304]);   /* 4 MB for dequantized f32 */
    /* 64K vocab slots (256 KB). TinyStories uses 32000; larger models
       need dynamic sizing. Cast from ptr[[u32; 65536]] to ptr[u32] so
       the tok_table parameter type matches the tokenizer helpers. */
    let tok_table = alloc([u32; 65536]) as ptr[u32];
    var tok_count: i64 = -1;
    let activation_buf = alloc([f32; 65536]) as ptr[f32];
    let kv_k_buf = alloc([f32; 442368]) as ptr[f32];   /* 6*256*288 */
    let kv_v_buf = alloc([f32; 442368]) as ptr[f32];
    let rope_freqs = alloc([f64; 24]) as ptr[f64];
    let logits_buf = alloc([f32; 32000]) as ptr[f32];
    let merge_table = alloc([u32; 65536]) as ptr[u32];
    var merge_count: i64 = -1;
    build_rope_freqs(rope_freqs, 48);
    println("[7] Enabling interrupts");
    interrupts_init();
    println("    timer at 100 Hz, keyboard ready");
    println("    commands: modules, info, config, tensors, embed, forward, fwdpos, generate [T] [K], encode, genprompt, tensor, matvec, decodetok, vocab, help");
    println("");

    var line_buf = alloc([u8; 128]) as ptr[u8];
    var line_len: i64 = 0;

    print("lepton> ");
    while (true) {
        cpu_hlt();
        let kbdr = kbd_read();
        if (kbdr != 0) {
            let ch = kbdr as u8;
            if (ch == 8 as u8 || ch == 127 as u8) {
                if (line_len > 0) {
                    line_len = line_len - 1;
                }
            } else if (ch == 10 as u8 || ch == 13 as u8) {
                let text = read_str(line_buf, 0, line_len);
                if (starts_with(text, "modules")) {
                    let n = mb_module_count();
                    println("  {n} module(s)");
                    for i in 0..(n as i64) {
                        let mbst = mb_module_start(i as u32);
                        let mbsz = mb_module_size(i as u32);
                        println("  [{i}] start={mbst} size={mbsz}");
                    }
                } else if (starts_with(text, "info")) {
                    if (mb_module_count() == 0 as u32) {
                        println("  no module loaded");
                    } else {
                        let model = mb_module_start(0 as u32) as ptr[u8];
                        let magic = read_u32(model, 0);
                        if (magic != 0x46554747 as u32) {
                            println("  not a GGUF file");
                        } else {
                            let ver = read_u32(model, 4);
                            let tc = read_u64(model, 8);
                            let kc = read_u64(model, 16);
                            println("  version = {ver}");
                            println("  tensors = {tc}");
                            println("  kv      = {kc}");
                        }
                    }
                } else if (starts_with(text, "tensor") && (text[6] == '\0' || text[6] == ' ')) {
                    if (mb_module_count() == 0 as u32) {
                        println("  no module loaded");
                    } else {
                        let model = mb_module_start(0 as u32) as ptr[u8];
                        let kc = read_u64(model, 16);
                        let kv_end = skip_kv_block(model, 24, kc);
                        let ti = parse_tensor_info(model, kv_end);
                        println("  name_len = {ti.name_len}");
                        println("  n_dims   = {ti.n_dims}");
                        println("  dims     = {ti.dim0} x {ti.dim1}");
                        println("  type     = {ti.ggml_type}");
                        println("  elements = {tensor_element_count(ti)}");
                        println("  bytes    = {tensor_byte_size(ti)}");
                    }
                } else if (starts_with(text, "matvec")) {
                    if (mb_module_count() == 0 as u32) {
                        println("  no module loaded");
                    } else {
                        let model = mb_module_start(0 as u32) as ptr[u8];
                        let tc = read_u64(model, 8);
                        let kc = read_u64(model, 16);
                        let kv_end = skip_kv_block(model, 24, kc);
                        var info_end = kv_end;
                        for i in 0..(tc as i64) {
                            info_end = info_end + tensor_info_size(model, info_end);
                        }
                        let rem = info_end % 32;
                        var data_off = info_end;
                        if (rem != 0) {
                            data_off = info_end + (32 - rem);
                        }

                        let ti = parse_tensor_info(model, kv_end);
                        let ty = ti.ggml_type;
                        let dim0 = ti.dim0;
                        let cols = ti.dim1;
                        let total = tensor_element_count(ti);
                        let src = (model + (data_off + (ti.data_offset as i64))) as ptr[u8];
                        let dst = scratch_bytes as ptr[f32];

                        if (ty == 2 as u32) {
                            let nc = cols as i64;
                            /* Only dequantize the first 8 rows. */
                            var rows_to_do: i64 = 8;
                            if (dim0 < 8 as u64) {
                                rows_to_do = dim0 as i64;
                            }
                            let nb = (rows_to_do * nc) / 32;
                            dequant_q4_0_tensor(src, dst, nb);
                            println("  Q4_0, {dim0}x{cols}, {total} elements, {nb} blocks of first {rows_to_do} rows");
                            for row in 0..rows_to_do {
                                var acc = 0.0f;
                                for col in 0..nc {
                                    acc = acc + dst[(row as i64) * nc + col];
                                }
                                println("  row {row} = {acc}");
                            }
                        } else if (ty == 8 as u32) {
                            let nc = cols as i64;
                            var rows_to_do: i64 = 8;
                            if (dim0 < 8 as u64) {
                                rows_to_do = dim0 as i64;
                            }
                            let nb = (rows_to_do * nc) / 32;
                            dequant_q8_0_tensor(src, dst, nb);
                            println("  Q8_0, {dim0}x{cols}, {total} elements, {nb} blocks of first {rows_to_do} rows");
                            for row in 0..rows_to_do {
                                var acc = 0.0f;
                                for col in 0..nc {
                                    acc = acc + dst[(row as i64) * nc + col];
                                }
                                println("  row {row} = {acc}");
                            }
                        } else {
                            println("  type={ty} not supported");
                        }
                    }
                } else if (starts_with(text, "decodetok")) {
                    if (mb_module_count() == 0 as u32) {
                        println("  no module loaded");
                    } else {
                        let tok_id = parse_int_at(text, 9);
                        if (tok_count < 0) {
                            tok_count = build_tok_offsets(
                                mb_module_start(0 as u32) as ptr[u8],
                                tok_table);
                            if (tok_count < 0) {
                                println("  no tokenizer.ggml.tokens in model");
                            } else {
                                println("  vocab size = {tok_count}");
                            }
                        }
                        if (tok_count > 0) {
                            if (tok_id < 0 || tok_id >= tok_count) {
                                println("  token id out of range");
                            } else {
                                let model = mb_module_start(0 as u32) as ptr[u8];
                                let tokstr = decode_tok(model, tok_table, tok_id as u32);
                                print("  [{tok_id}] '");
                                print_token(tokstr);
                                println("'");
                            }
                        }
                    }
                } else if (starts_with(text, "vocab")) {
                    if (mb_module_count() == 0 as u32) {
                        println("  no module loaded");
                    } else {
                        let a = parse_int_at(text, 5);
                        let a_end = parse_int_end(text, 5);
                        var b = parse_int_at(text, a_end);
                        if (b < a) { b = a; }
                        if (tok_count < 0) {
                            tok_count = build_tok_offsets(
                                mb_module_start(0 as u32) as ptr[u8],
                                tok_table);
                            if (tok_count < 0) {
                                println("  no tokenizer.ggml.tokens in model");
                            } else {
                                println("  vocab size = {tok_count}");
                            }
                        }
                        if (tok_count > 0) {
                            if (b >= tok_count) { b = tok_count - 1; }
                            if (a < 0) { a = 0; }
                            let model = mb_module_start(0 as u32) as ptr[u8];
                            for n in a..(b + 1) {
                                let tokstr = decode_tok(model, tok_table, n as u32);
                                print("  [{n}] '");
                                print_token(tokstr);
                                println("'");
                            }
                        }
                    }
                } else if (starts_with(text, "config")) {
                    if (mb_module_count() == 0 as u32) {
                        println("  no module loaded");
                    } else {
                        let cm = mb_module_start(0 as u32) as ptr[u8];
                        let ckc = read_u64(cm, 16);
                        let e1 = find_kv(cm, 24, ckc, "llama.embedding_length");
                        let e2 = find_kv(cm, 24, ckc, "llama.attention.head_count");
                        let e3 = find_kv(cm, 24, ckc, "llama.block_count");
                        let e4 = find_kv(cm, 24, ckc, "llama.feed_forward_length");
                        if (e1 >= 0) { println("  n_embd  = {read_u32(cm, e1)}"); }
                        if (e2 >= 0) { println("  n_head  = {read_u32(cm, e2)}"); }
                        if (e3 >= 0) { println("  n_layer = {read_u32(cm, e3)}"); }
                        if (e4 >= 0) { println("  n_ff    = {read_u32(cm, e4)}"); }
                    }
                } else if (starts_with(text, "tensors") && (text[7] == '\0' || text[7] == ' ')) {
                    if (mb_module_count() == 0 as u32) {
                        println("  no module loaded");
                    } else {
                        let tm = mb_module_start(0 as u32) as ptr[u8];
                        let ttc = read_u64(tm, 8);
                        let tkc = read_u64(tm, 16);
                        let tstart = skip_kv_block(tm, 24, tkc);
                        var twalk: i64 = tstart;
                        var tidx: i64 = 0;
                        while (tidx < (ttc as i64)) {
                            let ti2 = parse_tensor_info(tm, twalk);
                            let nm = read_str(tm, ti2.name_offset, ti2.name_len as i64);
                            println("  [{tidx}] '{nm}' {ti2.dim0}x{ti2.dim1} type={ti2.ggml_type}");
                            twalk = twalk + tensor_info_size(tm, twalk);
                            tidx = tidx + 1;
                        }
                    }
                } else if (starts_with(text, "embed")) {
                    if (mb_module_count() == 0 as u32) {
                        println("  no module loaded");
                    } else {
                        if (tok_count < 0) {
                            tok_count = build_tok_offsets(
                                mb_module_start(0 as u32) as ptr[u8],
                                tok_table);
                        }
                        let etok = parse_int_at(text, 5);
                        let em = mb_module_start(0 as u32) as ptr[u8];
                        let etc = read_u64(em, 8);
                        let ekc = read_u64(em, 16);
                        let estart = skip_kv_block(em, 24, ekc);
                        var ewalk: i64 = estart;
                        var eidx: i64 = 0;
                        while (eidx < (etc as i64)) {
                            ewalk = ewalk + tensor_info_size(em, ewalk);
                            eidx = eidx + 1;
                        }
                        let erem = ewalk % 32;
                        var edata: i64 = ewalk;
                        if (erem != 0) {
                            edata = ewalk + (32 - erem);
                        }
                        let emb = tensor_by_name(em, estart, etc, "token_embd.weight");
                        if (emb.name_len == 0 as u64) {
                            println("  token_embd.weight not found");
                        } else {
                            let edim = emb.dim0 as i64;
                            let eblocks = edim / 32;
                            let erow_off = edata + (emb.data_offset as i64)
                                         + etok * eblocks * 18;
                            let erow = scratch_bytes as ptr[f32];
                            dequant_q4_0_tensor((em + erow_off) as ptr[u8],
                                                erow, eblocks);
                            println("  token {etok}: '{decode_tok(em, tok_table, etok as u32)}'");
                            println("  d[0]={erow[0]} d[1]={erow[1]} d[2]={erow[2]} d[3]={erow[3]}");
                        }
                    }
                } else if (starts_with(text, "forward")) {
                    if (mb_module_count() == 0 as u32) {
                        println("  no module loaded");
                    } else {
                        if (tok_count < 0) {
                            tok_count = build_tok_offsets(
                                mb_module_start(0 as u32) as ptr[u8],
                                tok_table);
                        }
                        let fwd_model = mb_module_start(0 as u32) as ptr[u8];
                        let fwd_buf = scratch_bytes as ptr[f32];
                        let fwd_out = alloc([u32; 1]) as ptr[u32];
                        println("  running 6-layer forward for token <s> (id 1)...");
                        forward_token(fwd_model, 1, 6,
                                      fwd_buf, activation_buf,
                                      fwd_out, true);
                        println("  predicted token = {fwd_out[0]}");
                        let fwd_str = decode_tok(fwd_model, tok_table, fwd_out[0]);
                        print("  '");
                        print_token(fwd_str);
                        println("'");
                    }
                } else if (starts_with(text, "fwdpos")) {
                    if (mb_module_count() == 0 as u32) {
                        println("  no module loaded");
                    } else {
                        if (tok_count < 0) {
                            tok_count = build_tok_offsets(
                                mb_module_start(0 as u32) as ptr[u8],
                                tok_table);
                        }
                        let fp_pos = parse_int_at(text, 6);
                        let fp_end = parse_int_end(text, 6);
                        let fp_tok = parse_int_at(text, fp_end);
                        let fp_m = mb_module_start(0 as u32) as ptr[u8];
                        let fp_w = scratch_bytes as ptr[f32];
                        let fp_a = activation_buf as ptr[f32];
                        for ci in 0..(6 * 256 * 288) {
                            kv_k_buf[ci] = 0.0f;
                            kv_v_buf[ci] = 0.0f;
                        }
                        forward_token_at_pos(fp_m, fp_tok, fp_pos,
                                             6, 6, 48, fp_w, fp_a,
                                             kv_k_buf, kv_v_buf, rope_freqs,
                                             logits_buf);
                        let fp_best = lp_argmax(logits_buf, 32000 as i32);
                        println("  pos={fp_pos} token={fp_tok} -> {fp_best}");
                        let fp_s = decode_tok(fp_m, tok_table, fp_best);
                        print("  '");
                        print_token(fp_s);
                        println("'");
                    }
                } else if (starts_with(text, "generate") && (text[8] == '\0' || text[8] == ' ')) {
                    if (mb_module_count() == 0 as u32) {
                        println("  no module loaded");
                    } else {
                        if (tok_count < 0) {
                            tok_count = build_tok_offsets(
                                mb_module_start(0 as u32) as ptr[u8],
                                tok_table);
                        }
                        var gen_temp: f32 = 1.0f;
                        var gen_k: i64 = 40;
                        if (text[8] == ' ') {
                            gen_temp = parse_float_at(text, 9);
                            let tend = parse_float_end(text, 9);
                            let k_arg = parse_int_at(text, tend);
                            if (k_arg > 0) { gen_k = k_arg; }
                        }
                        lp_rand_seed(ticks() + 1 as u32);
                        let gen_m = mb_module_start(0 as u32) as ptr[u8];

                        println("  Model starts at RAM address: {mb_module_start(0 as u32)}");

                        let gen_w = scratch_bytes as ptr[f32];
                        let gen_a = activation_buf as ptr[f32];
                        for ci in 0..(6 * 256 * 288) {
                            kv_k_buf[ci] = 0.0f;
                            kv_v_buf[ci] = 0.0f;
                        }

                        println("  generating 30 tokens (temp={gen_temp}, top_k={gen_k})...");
                        print("  [dbg: start] ");

                        var cur: u32 = 1 as u32;
                        for step in 0..30 {
                            forward_token_at_pos(gen_m, cur as i64, step,
                                                 6, 6, 48, gen_w, gen_a,
                                                 kv_k_buf, kv_v_buf, rope_freqs,
                                                 logits_buf);
                            cur = lp_sample(logits_buf, 32000 as i32,
                                            gen_temp, gen_k as i32);
                            let gs = decode_tok(gen_m, tok_table, cur);
                            print_token(gs);
                        }
                        println("");
                    }
                } else if (starts_with(text, "encode") && (text[6] == '\0' || text[6] == ' ')) {
                    if (mb_module_count() == 0 as u32) {
                        println("  no module loaded");
                    } else {
                        if (tok_count < 0) {
                            tok_count = build_tok_offsets(
                                mb_module_start(0 as u32) as ptr[u8],
                                tok_table);
                        }
                        if (merge_count < 0) {
                            merge_count = build_merge_offsets(
                                mb_module_start(0 as u32) as ptr[u8],
                                merge_table);
                        }
                        if (merge_count < 0) {
                            merge_count = 0;   /* use greedy fallback */
                        }
                        {
                            let em = mb_module_start(0 as u32) as ptr[u8];
                            /* Line is at line_buf[0..line_len]. Text starts at 7. */
                            let enc_ids = alloc([u32; 256]) as ptr[u32];
                            let n_enc = encode_prompt(em, tok_table, tok_count,
                                                       merge_table, merge_count,
                                                       line_buf, 7 as i64,
                                                       line_len,
                                                       enc_ids);
                            println("  {n_enc} tokens:");
                            var ti: i64 = 0;
                            while (ti < n_enc) {
                                let tid = enc_ids[ti];
                                let dec = decode_tok(em, tok_table, tid);
                                print("  [{ti}] {tid} = '");
                                print_token(dec);
                                println("'");
                                ti = ti + 1;
                            }
                        }
                    }
                } else if (starts_with(text, "genprompt") && (text[9] == '\0' || text[9] == ' ')) {
                    if (mb_module_count() == 0 as u32) {
                        println("  no module loaded");
                    } else {
                        if (tok_count < 0) {
                            tok_count = build_tok_offsets(
                                mb_module_start(0 as u32) as ptr[u8],
                                tok_table);
                        }
                        if (merge_count < 0) {
                            merge_count = build_merge_offsets(
                                mb_module_start(0 as u32) as ptr[u8],
                                merge_table);
                        }
                        if (merge_count < 0) {
                            merge_count = 0;   /* use greedy fallback */
                        }
                        {
                            let gp_buf = alloc([u8; 512]) as ptr[u8];
                            let gp_end = parse_quoted(text, 9, gp_buf);
                            if (gp_end < 0) {
                                println("  usage: genprompt \"text\" [temp] [top_k]");
                            } else {
                                var gp_t: f64 = 1.0;
                                var gp_k: i64 = 40;
                                gp_t = parse_float_at(text, gp_end) as f64;
                                let gp_tend = parse_float_end(text, gp_end);
                                let gp_kval = parse_int_at(text, gp_tend);
                                if (gp_kval > 0) { gp_k = gp_kval; }
                                if (gp_t <= 0.0) { gp_t = 1.0; }
                                let gp_m = mb_module_start(0 as u32) as ptr[u8];
                                let gp_w = scratch_bytes as ptr[f32];
                                let gp_a = activation_buf as ptr[f32];
                                let gp_ids = alloc([u32; 256]) as ptr[u32];

                                /* Encode prompt */
                                var gp_len: i64 = 0;
                                var gp_i: i64 = 0;
                                while (gp_buf[gp_i] != 0 as u8) { gp_i = gp_i + 1; }
                                gp_len = encode_prompt(gp_m, tok_table, tok_count,
                                                       merge_table, merge_count,
                                                       gp_buf, 0 as i64, gp_i,
                                                       gp_ids);
                                println("  prompt = {gp_len} tokens");
                                /* Echo back what we decoded, to check roundtrip */
                                print("  enc: ");
                                var ei: i64 = 0;
                                while (ei < gp_len) {
                                    print("{gp_ids[ei]} ");
                                    ei = ei + 1;
                                }
                                println("");

                                /* Clear KV */
                                for ci in 0..(6 * 256 * 288) {
                                    kv_k_buf[ci] = 0.0f;
                                    kv_v_buf[ci] = 0.0f;
                                }
                                lp_rand_seed(ticks() + 1 as u32);

                                /* Run forward on prompt tokens */
                                var gp_pos: i64 = 0;
                                while (gp_pos < gp_len) {
                                    forward_token_at_pos(gp_m, gp_ids[gp_pos] as i64, gp_pos,
                                                         6, 6, 48, gp_w, gp_a,
                                                         kv_k_buf, kv_v_buf, rope_freqs,
                                                         logits_buf);
                                    gp_pos = gp_pos + 1;
                                }

                                /* Generate */
                                println("  generating 30 tokens (temp={gp_t}, top_k={gp_k})...");
                                print("  ");
                                var gp_next: u32 = lp_sample(logits_buf, 32000 as i32,
                                                              gp_t as f32, gp_k as i32);
                                let gp_s0 = decode_tok(gp_m, tok_table, gp_next);
                                print_token(gp_s0);

                                var gp_g: i64 = 1;
                                while (gp_g < 30) {
                                    forward_token_at_pos(gp_m, gp_next as i64, gp_pos,
                                                         6, 6, 48, gp_w, gp_a,
                                                         kv_k_buf, kv_v_buf, rope_freqs,
                                                         logits_buf);
                                    gp_next = lp_sample(logits_buf, 32000 as i32,
                                                        gp_t as f32, gp_k as i32);
                                    let gp_sn = decode_tok(gp_m, tok_table, gp_next);
                                    print_token(gp_sn);
                                    gp_pos = gp_pos + 1;
                                    gp_g = gp_g + 1;
                                }
                                println("");
                            }
                        }
                    }
                } else if (starts_with(text, "help")) {
                    println("    commands: modules, info, config, tensors, embed, forward, fwdpos, generate [T] [K], encode, genprompt, tensor, matvec, decodetok, vocab, help");
                } else if (line_len > 0) {
                    println("  {text}");
                }
                line_len = 0;
                print("lepton> ");
            }
            else if (line_len < 127) {
                line_buf[line_len] = ch;
                line_len = line_len + 1;
            }
        }
    }

}
