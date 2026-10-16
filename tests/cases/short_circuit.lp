fn main() {
    let a = 0;
    let b = 5;

    if (a != 0 && b / a > 2) {
        println("bad");
    } else {
        println("and ok");
    }

    let c = 5;
    let d = 0;

    if (c == 5 || d / c > 0) {
        println("or ok");
    } else {
        println("or bad");
    }
}
