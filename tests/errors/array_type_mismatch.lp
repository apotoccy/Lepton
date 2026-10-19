fn take(a: [f32; 4]) -> f32 {
    return a[0];
}

fn main() {
    var a: [i32; 4];
    a[0] = 1;
    println("{take(a)}");
}
