fn main() {
    let a = alloc([i32; 4]);
    a[0] = 10;
    a[1] = 20;
    a[2] = 30;
    a[3] = 40;
    println("a = {a[0]} {a[1]} {a[2]} {a[3]}");
}
