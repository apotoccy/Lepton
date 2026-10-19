fn main() {
    let b = alloc([u8; 8]);
    b[0] = 0xEF as u8;  b[1] = 0xBE as u8;
    b[2] = 0xAD as u8;  b[3] = 0xDE as u8;
    b[4] = 0xF0 as u8;  b[5] = 0xDE as u8;
    b[6] = 0xBC as u8;  b[7] = 0x9A as u8;

    println("u8  = {read_u8(b, 0)}");
    println("u16 = {read_u16(b, 0)}");
    println("u32 = {read_u32(b, 0)}");
    println("u64 = {read_u64(b, 0)}");
}
