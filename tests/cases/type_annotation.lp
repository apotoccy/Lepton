fn sum4(a: [f32; 4]) -> f32 {
    return a[0] + a[1] + a[2] + a[3];
}

fn add64(a: i64, b: i64) -> i64 {
    return a + b;
}

fn sign(x: i32) -> i32 {
    if (x > 0) {
        return 1;
    } else {
        return 0;
    }
}

fn one() -> i64 {
    return 1;
}

fn main() {
    let narrowed: i32 = 1.9;
    let v: v4f32 = [1, 2, 3, 4];
    let w = v + 0.5f;
    let a: [f32; 4] = {1, 2, 3, 4};
    let p: ptr[u8] = 0;
    var x: i32;
    x = 5;

    println("narrowed = {narrowed}");
    println("w = {w[0]} {w[1]} {w[2]} {w[3]}");
    println("sum = {sum4(a)}");
    println("sumlit = {sum4({1, 2, 3, 4})}");
    println("add64 = {add64(2, 3)}");
    println("sign = {sign(3)}");
    println("one = {one()}");
    println("p = {p}");
    println("x = {x}");
}
