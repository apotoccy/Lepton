enum E {
    A,
    B,
    C
}

fn main() {
    let x = E.B;
    if (x > E.A) {
        println("gt");
    }
    if (x < E.C) {
        println("lt");
    }
}
