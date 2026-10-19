import "std/nn/norm.lp";

fn main() {
    let x = [3.0f, 4.0f, 0.0f, 0.0f];
    let y = rmsnorm(x);
    println("y = {y[0]} {y[1]} {y[2]} {y[3]}");
}
