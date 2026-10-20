fn main() {
    let p = alloc([i32; 4]);
    p[0] = 7;
    println("{p[0]}");
    free(p);
    alloc_reset();
    let q = alloc([i32; 4]);
    q[0] = 9;
    println("{q[0]}");
}
