fn main() {
    let buf = alloc([u8; 8]);
    buf[0] = 1;
    buf[1] = 2;
    buf[2] = 3;
    buf[3] = 4;
    buf[4] = 5;
    buf[5] = 6;
    buf[6] = 7;
    buf[7] = 8;

    write_file("nums.bin", buf, 8);

    let chunk = alloc([u8; 4]);
    read_file_at("nums.bin", 2, chunk, 4);
    println("chunk = {chunk[0]} {chunk[1]} {chunk[2]} {chunk[3]}");
}
