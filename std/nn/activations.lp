/* Neural-network activations. Written in Lepton, no libm. */

fn gelu(x: f32) -> f32 {
    /* Tanh approximation, Hendrycks & Gimpel 2016. */
    let c = 0.7978845608f;                 /* sqrt(2 / pi) */
    let inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanh(inner));
}

fn gelu_vec(v: v4f32) -> v4f32 {
    let c = 0.7978845608f;
    let x3 = v * v * v;
    let inner = c * (v + 0.044715f * x3);
    let t = tanh(inner);
    return 0.5f * v * (1.0f + t);
}

fn relu(x: f32) -> f32 {
    if (x > 0.0f) {
        return x;
    }
    return 0.0f;
}

fn relu_vec(v: v4f32) -> v4f32 {
    let z = [0.0f, 0.0f, 0.0f, 0.0f];
    return vmax2(v, z);
}

fn relu_vec_masked(v: v4f32) -> v4f32 {
    let z = [0.0f, 0.0f, 0.0f, 0.0f];
    let mask = vcmpgt(v, z);
    return vselect(mask, v, z);
}
