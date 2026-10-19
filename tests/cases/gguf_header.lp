import "std/gguf/header.lp";

fn main() {
    /* Hand-craft a 24-byte GGUF v3 header with tensor_count=5, kv_count=2. */
    let b = alloc([u8; 24]);

    /* magic: "GGUF" */
    b[0]  = 71 as u8;  /* G */
    b[1]  = 71 as u8;  /* G */
    b[2]  = 85 as u8;  /* U */
    b[3]  = 70 as u8;  /* F */

    /* version: 3 (LE u32) */
    b[4] = 3 as u8;
    b[5] = 0 as u8;
    b[6] = 0 as u8;
    b[7] = 0 as u8;

    /* tensor_count: 5 (LE u64, low byte only) */
    b[8] = 5 as u8;
    b[9] = 0 as u8;  b[10] = 0 as u8;  b[11] = 0 as u8;
    b[12] = 0 as u8; b[13] = 0 as u8;  b[14] = 0 as u8;  b[15] = 0 as u8;

    /* kv_count: 2 (LE u64, low byte only) */
    b[16] = 2 as u8;
    b[17] = 0 as u8; b[18] = 0 as u8;  b[19] = 0 as u8;
    b[20] = 0 as u8; b[21] = 0 as u8;  b[22] = 0 as u8;  b[23] = 0 as u8;

    write_file("t.gguf", b, 24);

    let h = parse_header("t.gguf");
    println("version = {h.version}");
    println("tensors = {h.tensor_count}");
    println("kv      = {h.kv_count}");
}
