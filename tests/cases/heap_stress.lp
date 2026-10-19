fn main() {
    let big = alloc([i32; 1000]);
    var sum = 0;
    for i in 0..1000 {
        big[i] = i;
    }
    for i in 0..1000 {
        sum = sum + big[i];
    }
    println("sum = {sum}");
}
