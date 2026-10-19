/* Dequantization for GGUF block-quantized formats.

   Q4_0 block: 18 bytes, 32 elements.
     [0..1]  d       f16 scale
     [2..17] qs      16 bytes; each byte packs 2 weights as 4-bit signed
                     values in [-8, 7] (stored +8 biased, so raw 0..15).

   The element mapping is NOT interleaved: the low nibble of qs[j] becomes
   element j (0..15), the high nibble becomes element j+16 (16..31). This
   matches ggml's reference unpacking.

   Q8_0 block: 34 bytes, 32 elements.
     [0..1]  d       f16 scale
     [2..33] qs      32 bytes; each is a signed int8 in [-128, 127].

   Element mapping is direct: qs[i] is element i.

   Dequantized value in both cases is: d * (signed weight). */

fn dequant_q4_0_block(src: ptr[u8], dst: ptr[f32]) -> void {
    let d = read_f16(src, 0);
    let qs = src + 2;
    for j in 0..16 {
        let byte = qs[j];
        let lo = (byte & 0x0F as u8) as i32;
        let hi = ((byte >> 4 as u8) & 0x0F as u8) as i32;
        dst[j]      = d * ((lo - 8) as f32);
        dst[j + 16] = d * ((hi - 8) as f32);
    }
}

fn dequant_q8_0_block(src: ptr[u8], dst: ptr[f32]) -> void {
    let d = read_f16(src, 0);
    let qs = (src + 2) as ptr[i8];
    for i in 0..32 {
        dst[i] = d * ((qs[i] as i32) as f32);
    }
}

/* Dequantize a whole tensor of Q4_0 blocks. `n_blocks` is
   tensor_element_count(t) / 32, computed by the caller from the
   tensor info struct. `dst` must hold 32 * n_blocks f32 elements. */
fn dequant_q4_0_tensor(src: ptr[u8], dst: ptr[f32], n_blocks: i64) -> void {
    for i in 0..n_blocks {
        dequant_q4_0_block(src + i * 18, dst + i * 32);
    }
}

fn dequant_q8_0_tensor(src: ptr[u8], dst: ptr[f32], n_blocks: i64) -> void {
    for i in 0..n_blocks {
        dequant_q8_0_block(src + i * 34, dst + i * 32);
    }
}
