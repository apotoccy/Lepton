fn main() {
    let a = "alpha";
    let b = "alpha";
    let c = "beta";
    println("{a == b} {a != c} {a < c} {c > a} {a <= b} {a >= c}");
    if (a == "alpha") {
        println("match");
    }
    let n = 0 as str;
    println("{n == n} {n == a} {n < a}");
}
