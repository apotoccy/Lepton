struct P { x: i32; }
fn make() -> P {
    return P(x: 1);
}
fn main() {
    make().x = 2;
}
