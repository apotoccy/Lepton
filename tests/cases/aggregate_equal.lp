struct Point {
    x: i32;
    y: i32;
}

struct Box {
    p: Point;
    data: [i32; 3];
    name: str;
    vec: v4f32;
}

struct F {
    x: f32;
}

fn main() {
    let a = Point(x: 1, y: 2);
    let b = Point(x: 1, y: 2);
    let c = Point(x: 1, y: 3);
    println("{a == b} {a != b} {a == c}");

    let x = Box(p: a, data: {1, 2, 3}, name: "hi",
                vec: [1.0f, 2.0f, 3.0f, 4.0f]);
    let y = Box(p: b, data: {1, 2, 3}, name: "hi",
                vec: [1.0f, 2.0f, 3.0f, 4.0f]);
    let z = Box(p: b, data: {1, 2, 4}, name: "hi",
                vec: [1.0f, 2.0f, 3.0f, 4.0f]);
    println("{x == y} {x != y} {x == z}");

    let a1: [Point; 2] = {a, b};
    let a2: [Point; 2] = {b, a};
    let a3: [Point; 2] = {a, c};
    println("{a1 == a2} {a1 != a2} {a1 == a3}");

    let nan = 0.0f / 0.0f;
    let n1 = F(x: nan);
    let n2 = F(x: nan);
    println("{n1 == n2} {n1 != n2}");
}
