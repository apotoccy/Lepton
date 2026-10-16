fn main() {
    var sum = 0;
    for i in 0..10 {
        if (i == 5) {
            break;
        }
        sum = sum + i;
    }
    if (sum == 10) {
        println("break ok");
    } else {
        println("break bad");
    }

    var sum2 = 0;
    for i in 0..6 {
        if (i == 2) {
            continue;
        }
        sum2 = sum2 + i;
    }
    if (sum2 == 13) {
        println("continue ok");
    } else {
        println("continue bad");
    }
}
