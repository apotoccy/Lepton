/* Normalisation primitives. */

fn rmsnorm(x: v4f32) -> v4f32 {
    /* Root-mean-square normalisation over a 4-element vector. */
    let ms = sum(x * x) / 4.0f;
    let inv = rsqrt(ms + 1e-6f);
    return x * inv;
}

fn softmax(x: v4f32) -> v4f32 {
    /* Numerically stable softmax: subtract the max before exp. */
    let m = vmax(x);
    let e = exp(x - m);
    let s = sum(e);
    return e / s;
}
