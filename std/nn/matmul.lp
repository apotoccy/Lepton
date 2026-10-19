/* Matrix multiplication for the 4×4 case, the innermost kernel of every
   attention block. Row-major flat arrays, passed by reference. */

fn matmul4(a: ptr[[f32; 16]], b: ptr[[f32; 16]]) -> [f32; 16] {
    var c: [f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            var s = 0.0f;
            for k in 0..4 {
                s = s + a[i * 4 + k] * b[k * 4 + j];
            }
            c[i * 4 + j] = s;
        }
    }
    return c;
}

fn matvec4(m: ptr[[f32; 16]], v: v4f32) -> v4f32 {
    var r = [0.0f, 0.0f, 0.0f, 0.0f];
    for i in 0..4 {
        let s = m[i * 4 + 0] * v[0]
              + m[i * 4 + 1] * v[1]
              + m[i * 4 + 2] * v[2]
              + m[i * 4 + 3] * v[3];
        r[i] = s;
    }
    return r;
}

fn identity4() -> [f32; 16] {
    return {1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f};
}

fn add_inplace(a: ptr[[f32; 16]], b: ptr[[f32; 16]]) -> void {
    for i in 0..16 {
        a[i] = a[i] + b[i];
    }
}
