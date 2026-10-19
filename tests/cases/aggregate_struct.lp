enum Kind {
    A,
    B
}

struct Item {
    data: [f32; 4];
    kind: Kind;
    n: i32;
}

fn main() {
    let x = Item(data: {1.0f, 2.0f, 3.0f, 4.0f}, kind: Kind.B, n: 4);
    println("{x.data[0]} {x.data[3]} {x.n}");
    if (x.kind == Kind.B) {
        println("kind is B");
    }
}
