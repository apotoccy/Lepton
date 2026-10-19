import "std/nn/matmul.lp";

fn main() {
    let a = alloc([f32; 16]);
    let b = alloc([f32; 16]);
    let i = alloc([f32; 16]);

    for k in 0..16 {
        a[k] = 0.0f;
        b[k] = 0.0f;
        i[k] = 0.0f;
    }

    a[0] = 1.0f; a[1] = 2.0f; a[4] = 3.0f; a[5] = 4.0f;
    b[0] = 5.0f; b[1] = 6.0f; b[4] = 7.0f; b[5] = 8.0f;

    let c = matmul4(a, b);

    println("c[0] = {c[0]}  c[1] = {c[1]}");
    println("c[4] = {c[4]}  c[5] = {c[5]}");
}
