fn main() {
    let a = alloc([u8; 4]);
    let b = a;
    if (a == b) {
        println("same");
    }
    if (a != 0 as ptr[[u8; 4]]) {
        println("not null");
    }
}
