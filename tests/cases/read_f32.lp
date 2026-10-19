fn main() {
    let b = alloc([u8; 4]);

    b[0] = 0 as u8;  b[1] = 0 as u8;
    b[2] = 0x80 as u8;  b[3] = 0x3F as u8;
    println("one = {read_f32(b, 0)}");

    b[0] = 0 as u8;  b[1] = 0 as u8;
    b[2] = 0x20 as u8;  b[3] = 0xC0 as u8;
    println("neg = {read_f32(b, 0)}");

    b[0] = 0xFF as u8;  b[1] = 0xFF as u8;
    b[2] = 0xFF as u8;  b[3] = 0xFF as u8;
    println("i32 = {read_i32(b, 0)}");
}
