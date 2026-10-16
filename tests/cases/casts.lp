fn main() {
    let i: i32 = 42;
    let f = i as f16;
    let g = f as f32;
    let j = g as i32;
    let t = 3.9 as i32;

    println("i = {i}");
    println("f = {f}");
    println("g = {g}");
    println("j = {j}");
    println("t = {t}");
}
