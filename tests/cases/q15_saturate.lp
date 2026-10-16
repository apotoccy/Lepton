fn main() {
    let big = 0.75 as q15;
    let sum = big + big;
    let neg = (-1.0) as q15;

    println("big = {big}");
    println("sum = {sum}");
    println("neg = {neg}");

    let x = 0.75 as q15;
    let y = 0.5 as q15;
    let z = x + y;

    if (z > x) {
        println("grew");
    } else {
        println("no growth");
    }
}
