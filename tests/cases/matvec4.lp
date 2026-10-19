import "std/nn/matmul.lp";

fn main() {
    let m = {1.0f, 0.0f, 0.0f, 0.0f,
             0.0f, 2.0f, 0.0f, 0.0f,
             0.0f, 0.0f, 3.0f, 0.0f,
             0.0f, 0.0f, 0.0f, 4.0f};
    let v = [1.0f, 1.0f, 1.0f, 1.0f];
    let r = matvec4(&m, v);
    println("r = {r[0]} {r[1]} {r[2]} {r[3]}");
}
