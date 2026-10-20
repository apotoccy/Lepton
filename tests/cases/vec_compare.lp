fn v4(x: v4i32) -> void {
    println("{x[0]} {x[1]} {x[2]} {x[3]}");
}

fn f4(x: v4f32) -> void {
    println("{x[0]} {x[1]} {x[2]} {x[3]}");
}

fn main() {
    let a: v4i32 = [1, 2, 3, 4];
    let b: v4i32 = [2, 2, 2, 5];
    v4(vselect(vcmpeq(a, b), a, b));
    v4(vselect(vcmpne(a, b), a, b));
    v4(vselect(vcmplt(a, b), a, b));
    v4(vselect(vcmple(a, b), a, b));
    v4(vselect(vcmpgt(a, b), a, b));
    v4(vselect(vcmpge(a, b), a, b));
    v4(vmax2(a, b));
    v4(vmin2(a, b));

    let x: v4f32 = [1.0f, 2.0f, 3.0f, 4.0f];
    let y: v4f32 = [2.0f, 2.0f, 2.0f, 5.0f];
    f4(vselect(vcmpeq(x, y), x, y));
    f4(vselect(vcmpne(x, y), x, y));
    f4(vselect(vcmplt(x, y), x, y));
    f4(vselect(vcmple(x, y), x, y));
    f4(vselect(vcmpgt(x, y), x, y));
    f4(vselect(vcmpge(x, y), x, y));
    f4(vmax2(x, y));
    f4(vmin2(x, y));
}
