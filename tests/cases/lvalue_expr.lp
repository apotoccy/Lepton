struct Point {
    x: i32;
    y: i32;
}

fn main() {
    let buf = alloc([i32; 4]);
    buf[0] = 1;
    buf[1] = 2;
    buf[2] = 3;
    buf[3] = 4;
    let p = buf as ptr[i32];
    (p + 1)[0] = 7;
    (p + 2)[0] += 3;
    println("{buf[0]} {buf[1]} {buf[2]} {buf[3]}");

    var a: [i32; 3] = {1, 2, 3};
    (&a)[1] = 9;
    (&a)[0] += 4;
    println("{a[0]} {a[1]} {a[2]}");

    var s = Point(x: 1, y: 2);
    (&s)[0].x = 7;
    (&s)[0].y += 3;
    println("{s.x} {s.y}");

    let q = buf as ptr[i32];
    (q + 3)[0] = 11;
    println("{buf[3]}");
}
