fn main() {
    let fbuf = alloc([f32; 1]);
    fbuf[0] = 1.5f;
    write_file("f.bin", fbuf as ptr[[u8; 4]], 4);

    let bbuf = alloc([u8; 4]);
    read_file_at("f.bin", 0, bbuf, 4);
    let fback = bbuf as ptr[[f32; 1]];
    println("f = {fback[0]}");
}
