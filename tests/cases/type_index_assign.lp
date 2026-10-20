fn main() {
    var a: [f32; 4] = {0.0f, 0.0f, 0.0f, 0.0f};
    a[0] = 1;
    a[1] = 2.5;

    var v: v4f32 = [0.0f, 0.0f, 0.0f, 0.0f];
    v[0] = 1;
    v[1] = 2;

    println("a0={a[0]} a1={a[1]}");
    println("v0={v[0]} v1={v[1]}");
}
