fn main() {
    let a = alloc([u8; 4]);
    let b = alloc([i32; 4]);
    if (a == b) {
        println("same");
    }
}
