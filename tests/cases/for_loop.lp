fn main() {
    var sum = 0;
    for i in 0..5 {
        sum = sum + i;
    }
    if (sum == 10) {
        println("for ok");
    } else {
        println("for bad");
    }
}
