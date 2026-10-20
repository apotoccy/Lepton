fn main() {
    let a: [i32; 4] = {1, 2, 3, 4};
    let b = a as [f32; 4];
    let c = b as [i32; 4];
    println("{b[0]} {b[1]} {b[2]} {b[3]}");
    println("{c[0]} {c[1]} {c[2]} {c[3]}");
    let q = b as [q15; 4];
    println("{q[0]} {q[1]} {q[2]} {q[3]}");
}
