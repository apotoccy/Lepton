fn main() {
    let mark = alloc_mark();
    let p = alloc([i32; 2]);
    p[0] = 1;
    let q = alloc([i32; 2]);
    q[0] = 2;
    alloc_release(mark);
    println("released");
}
