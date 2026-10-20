struct P {
    x: i32;
}

fn main() {
    let a = P(x: 1);
    let b = P(x: 1);
    if (a < b) {
        println("same");
    }
}
