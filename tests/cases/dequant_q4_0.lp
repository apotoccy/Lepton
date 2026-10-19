import "std/gguf/dequant.lp";

fn main() {
    /* Build one Q4_0 block with d = 0.5, qs[j] = j * 17 (0x00..0xFF). */
    let block = alloc([u8; 18]);

    /* f16 for 0.5 = 0x3800, LE bytes: 0x00, 0x38. */
    block[0] = 0x00 as u8;
    block[1] = 0x38 as u8;

    for j in 0..16 {
        block[2 + j] = ((j * 17) as u8);
    }

    let out = alloc([f32; 32]);
    dequant_q4_0_block(block as ptr[u8], out as ptr[f32]);

    /* Low nibbles: qs[j] & 0x0F = j. Elements 0..15 = 0.5*(j-8). */
    println("e0  = {out[0]}");
    println("e1  = {out[1]}");
    println("e7  = {out[7]}");
    println("e8  = {out[8]}");
    println("e15 = {out[15]}");

    /* High nibbles: qs[j] >> 4 = j. Elements 16..31 mirror 0..15. */
    println("e16 = {out[16]}");
    println("e23 = {out[23]}");
    println("e31 = {out[31]}");
}
