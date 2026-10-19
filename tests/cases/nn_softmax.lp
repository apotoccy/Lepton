import "std/nn/norm.lp";

fn main() {
    let u = [0.0f, 0.0f, 0.0f, 0.0f];
    let s = softmax(u);
    println("s0 = {s[0]}");
    println("s1 = {s[1]}");
    println("s2 = {s[2]}");
    println("s3 = {s[3]}");
}
