import "std/gguf/kv.lp";

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

    /* GGUF header: 24 bytes. */
    put_u8(buf, 0,  71 as u8);   /* G */
    put_u8(buf, 1,  71 as u8);   /* G */
    put_u8(buf, 2,  85 as u8);   /* U */
    put_u8(buf, 3,  70 as u8);   /* F */
    put_u32(buf, 4, 3 as u32);   /* version */
    put_u64(buf, 8, 0 as u64);   /* tensor_count */
    put_u64(buf, 16, 3 as u64);  /* kv_count */

    /* KV 1: "general.architecture" (20 chars) -> "llama" (string). */
    var o = 24;
    put_u64(buf, o, 20 as u64); o = o + 8;
    o = put_bytes(buf, o, "general.architecture");
    put_u32(buf, o, 8 as u32); o = o + 4;  /* type 8 = string */
    put_u64(buf, o, 5 as u64); o = o + 8;
    o = put_bytes(buf, o, "llama");

    /* KV 2: "llama.context_length" (20 chars) -> 2048 (u32). */
    put_u64(buf, o, 20 as u64); o = o + 8;
    o = put_bytes(buf, o, "llama.context_length");
    put_u32(buf, o, 4 as u32); o = o + 4;  /* type 4 = u32 */
    put_u32(buf, o, 2048 as u32); o = o + 4;

    /* KV 3: "general.alignment" (17 chars) -> 32 (u32). */
    put_u64(buf, o, 17 as u64); o = o + 8;
    o = put_bytes(buf, o, "general.alignment");
    put_u32(buf, o, 4 as u32); o = o + 4;
    put_u32(buf, o, 32 as u32); o = o + 4;

    let total = o;
    write_file("t.gguf", buf, total);

    /* Re-read the KV block into a fresh buffer and parse. */
    let kv = alloc([u8; 262144]);
    read_file_at("t.gguf", 24, kv, 262144 as i64);

    let off = find_kv(kv, 0, 3 as u64, "llama.context_length");
    if (off < 0) {
        println("context_length: not found");
    } else {
        println("context_length = {read_u32(kv, off)}");
    }

    let off2 = find_kv(kv, 0, 3 as u64, "general.alignment");
    if (off2 < 0) {
        println("alignment: not found");
    } else {
        println("alignment = {read_u32(kv, off2)}");
    }

    let off3 = find_kv(kv, 0, 3 as u64, "missing.key");
    println("missing = {off3}");

    let end = skip_kv_block(kv, 0, 3 as u64);
    println("kv block size = {end}");
}
