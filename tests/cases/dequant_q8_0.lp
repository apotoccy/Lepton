import "std/gguf/dequant.lp";

fn main() {
    /* One Q8_0 block: d = 0.5, qs[i] = i - 16 (range -16..15). */
    let block = alloc([u8; 34]);
    block[0] = 0x00 as u8;
    block[1] = 0x38 as u8;   /* f16 for 0.5 */

    for i in 0..32 {
        block[2 + i] = (((i as i32) - 16) as i8) as u8;
    }

    let out = alloc([f32; 32]);
    dequant_q8_0_block(block as ptr[u8], out as ptr[f32]);

    /* out[i] = 0.5 * (i - 16) */
    println("e0  = {out[0]}");
    println("e16 = {out[16]}");
    println("e31 = {out[31]}");
}
