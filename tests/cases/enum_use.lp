enum Color {
    Red,
    Green,
    Blue
}

fn pick(c: Color) -> i32 {
    if (c == Color.Green) {
        return 2;
    }
    return 0;
}

fn main() {
    let a: Color = Color.Red;
    let b: Color = Color.Green;

    if (a == Color.Red) {
        println("a is red");
    } else {
        println("a is not red");
    }

    println("pick green = {pick(b)}");
}
