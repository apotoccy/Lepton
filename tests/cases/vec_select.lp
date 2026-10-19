fn main() {
    let a = [1.0f, -2.0f, 3.0f, -4.0f];
    let z = [0.0f, 0.0f, 0.0f, 0.0f];
    let mask = vcmpgt(a, z);
    let r = vselect(mask, a, z);
    println("r = {r[0]} {r[1]} {r[2]} {r[3]}");
}
