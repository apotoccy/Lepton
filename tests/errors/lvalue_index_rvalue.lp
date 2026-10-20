fn make() -> [i32; 2] {
    return {1, 2};
}

fn main() {
    make()[0] = 3;
}
