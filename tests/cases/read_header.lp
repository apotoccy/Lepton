fn main() {
    /* Hand-build a minimal GGUF header: magic "GGUF", version 3,
       tensor_count 0, kv_count 0. Total 24 bytes. */
    let b = alloc([u8; 24]);
    b[0] = 71 as u8;  b[1] = 71 as u8;
    b[2] = 85 as u8;  b[3] = 70 as u8;
    b[4] = 3 as u8;   b[5] = 0 as u8;
    b[6] = 0 as u8;   b[7] = 0 as u8;
    for i in 8..24 { b[i] = 0 as u8; }

    write_file("t.gguf", b, 24);
    let sz = file_size("t.gguf");
    println("size = {sz}");

    let rb = alloc([u8; 24]);
    read_file_at("t.gguf", 0, rb, 24);

    /* "GGUF" bytes in LE as u32: 0x46554747. */
    if (read_u32(rb, 0) == 0x46554747 as u32) {
        println("magic ok");
    } else {
        println("magic bad");
    }
    println("version = {read_u32(rb, 4)}");
    println("tensor_count = {read_u64(rb, 8)}");
}
