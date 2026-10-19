fn main() {
    let a = [1.0f, -2.0f, 3.0f, -4.0f];
    let b = [0.0f, 0.0f, 0.0f, 0.0f];
    let hi = vmax2(a, b);
    let lo = vmin2(a, b);
    println("hi = {hi[0]} {hi[1]} {hi[2]} {hi[3]}");
    println("lo = {lo[0]} {lo[1]} {lo[2]} {lo[3]}");
}
