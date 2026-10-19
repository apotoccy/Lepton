fn main() {
    var a: [i32; 4];
    for i in 0..4 {
        a[i] = i * i;
    }
    println("a = {a[0]} {a[1]} {a[2]} {a[3]}");
}
