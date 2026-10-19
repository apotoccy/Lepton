fn main() {
    /* Write 4 f32 values at offset 0 of a byte buffer. */
    let buf = alloc([u8; 20]);
    let floats = buf as ptr[[f32; 4]];
    floats[0] = 1.5f;
    floats[1] = 2.5f;
    floats[2] = 3.5f;
    floats[3] = 4.5f;

    /* Read them back starting at byte offset 4 (element 1). */
    let p = (buf as ptr[u8]) + 4;
    let f = p as ptr[[f32; 4]];
    println("f[0] = {f[0]}");
    println("f[1] = {f[1]}");
    println("f[2] = {f[2]}");
}
