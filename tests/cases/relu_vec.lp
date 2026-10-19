import "std/nn/activations.lp";

fn main() {
    let v = [-1.0f, 2.0f, -3.0f, 4.0f];
    let r = relu_vec(v);
    println("r = {r[0]} {r[1]} {r[2]} {r[3]}");

    let r2 = relu_vec_masked(v);
    println("r2 = {r2[0]} {r2[1]} {r2[2]} {r2[3]}");
}
