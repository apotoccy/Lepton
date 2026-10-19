fn main() {
    let buf = alloc([u8; 16]);
    buf[0] = 65;
    buf[1] = 66;
    buf[2] = 67;
    buf[3] = 68;

    let n = write_file("test.bin", buf, 4);
    println("wrote = {n}");

    let sz = file_size("test.bin");
    println("size = {sz}");

    let buf2 = alloc([u8; 16]);
    let m = read_file_at("test.bin", 0, buf2, 4);
    println("read = {m}");
    println("bytes = {buf2[0]} {buf2[1]} {buf2[2]} {buf2[3]}");
}
