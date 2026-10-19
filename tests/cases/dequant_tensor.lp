import "std/gguf/dequant.lp";

fn main() {
    /* Two Q8_0 blocks: block 0 has d = 0.25, block 1 has d = 0.5.
       Both blocks use qs[i] = 1 for all i, so all elements of block 0
       are 0.25 and all of block 1 are 0.5. */
    let buf = alloc([u8; 68]);

    /* Block 0 at offset 0: d = 0.25 (f16 = 0x3400), qs all 1. */
    buf[0] = 0x00 as u8;
    buf[1] = 0x34 as u8;
    for i in 0..32 { buf[2 + i] = 1 as u8; }

    /* Block 1 at offset 34: d = 0.5, qs all 1. */
    buf[34] = 0x00 as u8;
    buf[35] = 0x38 as u8;
    for i in 0..32 { buf[36 + i] = 1 as u8; }

    let out = alloc([f32; 64]);
    dequant_q8_0_tensor(buf as ptr[u8], out as ptr[f32], 2 as i64);

    println("b0[0]  = {out[0]}");
    println("b0[31] = {out[31]}");
    println("b1[0]  = {out[32]}");
    println("b1[31] = {out[63]}");
}
