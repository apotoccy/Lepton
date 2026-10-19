import "std/gguf/tensor.lp";

fn put_u8(b: ptr[[u8; 262144]], o: i64, v: u8) {
    b[o] = v;
}
fn put_u32(b: ptr[[u8; 262144]], o: i64, v: u32) {
    b[o]     = ( v        & 0xFF as u32) as u8;
    b[o + 1] = ((v >>  8) & 0xFF as u32) as u8;
    b[o + 2] = ((v >> 16) & 0xFF as u32) as u8;
    b[o + 3] = ((v >> 24) & 0xFF as u32) as u8;
}
fn put_u64(b: ptr[[u8; 262144]], o: i64, v: u64) {
    put_u32(b, o,     (v & 0xFFFFFFFF as u64) as u32);
    put_u32(b, o + 4, (v >> 32) as u32);
}
fn put_bytes(b: ptr[[u8; 262144]], o: i64, s: str) -> i64 {
    var i = 0;
    while (s[i] != '\0') {
        b[o + (i as i64)] = s[i] as u8;
        i = i + 1;
    }
    return o + (i as i64);
}

fn main() {
    let buf = alloc([u8; 262144]);

    /* header, 24 bytes, 3 tensors, no KV */
    put_u8(buf, 0, 71 as u8);   /* G */
    put_u8(buf, 1, 71 as u8);   /* G */
    put_u8(buf, 2, 85 as u8);   /* U */
    put_u8(buf, 3, 70 as u8);   /* F */
    put_u32(buf, 4, 3 as u32);  /* version */
    put_u64(buf, 8, 3 as u64);  /* tensor_count */
    put_u64(buf, 16, 0 as u64); /* kv_count */

    /* tensor 0: "token_embd.weight" (17 chars), 2D [32000, 4096], F32, offset 0 */
    var o = 24;
    put_u64(buf, o, 17 as u64); o = o + 8;
    o = put_bytes(buf, o, "token_embd.weight");
    put_u32(buf, o, 2 as u32); o = o + 4;
    put_u64(buf, o, 32000 as u64); o = o + 8;
    put_u64(buf, o, 4096 as u64); o = o + 8;
    put_u32(buf, o, 0 as u32); o = o + 4;         /* F32 */
    put_u64(buf, o, 0 as u64); o = o + 8;

    /* tensor 1: "output_norm.weight" (18 chars), 1D [4096], F32, offset 524288000 */
    put_u64(buf, o, 18 as u64); o = o + 8;
    o = put_bytes(buf, o, "output_norm.weight");
    put_u32(buf, o, 1 as u32); o = o + 4;
    put_u64(buf, o, 4096 as u64); o = o + 8;
    put_u32(buf, o, 0 as u32); o = o + 4;
    put_u64(buf, o, 524288000 as u64); o = o + 8;

    /* tensor 2: "blk.0.attn_q.weight" (20 chars), 2D [4096, 4096], Q8_0, offset 524304384 */
    put_u64(buf, o, 19 as u64); o = o + 8;
    o = put_bytes(buf, o, "blk.0.attn_q.weight");
    put_u32(buf, o, 2 as u32); o = o + 4;
    put_u64(buf, o, 4096 as u64); o = o + 8;
    put_u64(buf, o, 4096 as u64); o = o + 8;
    put_u32(buf, o, 8 as u32); o = o + 4;         /* Q8_0 */
    put_u64(buf, o, 524304384 as u64); o = o + 8;

    write_file("t.gguf", buf, o);

    /* read back */
    let tb = alloc([u8; 262144]);
    read_file_at("t.gguf", 24, tb, 262144 as i64);

    let t0 = find_tensor_by_index(tb, 0, 0 as u64);
    let t1 = find_tensor_by_index(tb, 0, 1 as u64);
    let t2 = find_tensor_by_index(tb, 0, 2 as u64);

    println("t0 name_len = {t0.name_len}");
    println("t0 dims = {t0.dim0} {t0.dim1} {t0.dim2} {t0.dim3}");
    println("t0 elements = {tensor_element_count(t0)}");
    println("t0 bytes = {tensor_byte_size(t0)}");

    println("t1 name_len = {t1.name_len}");
    println("t1 elements = {tensor_element_count(t1)}");
    println("t1 bytes = {tensor_byte_size(t1)}");

    println("t2 name_len = {t2.name_len}");
    println("t2 ggml_type = {t2.ggml_type}");
    println("t2 bytes = {tensor_byte_size(t2)}");
}
