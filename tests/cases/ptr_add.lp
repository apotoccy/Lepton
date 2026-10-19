fn main() {
    let buf = alloc([u8; 16]);
    for i in 0..16 {
        buf[i] = ((i * 2) as u8);
    }

    /* Cast the buffer to a scalar pointer and advance it. */
    let p = buf as ptr[u8];
    let q = p + 4;
    println("q[0] = {q[0]}");
    println("q[1] = {q[1]}");

    /* Element difference. */
    let r = p + 8;
    let delta = r - q;
    println("delta = {delta}");
}
