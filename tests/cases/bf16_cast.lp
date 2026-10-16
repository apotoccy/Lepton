fn main() {
    let x = 3.75 as bf16;
    let back = x as f32;
    let n = x as i32;

    println("x = {x}");
    println("back = {back}");
    println("n = {n}");

    if (x > (1.0 as bf16)) {
        println("bigger");
    } else {
        println("smaller");
    }
}
