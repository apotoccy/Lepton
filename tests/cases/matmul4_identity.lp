import "std/nn/matmul.lp";

fn main() {
    let a = {1.0f, 2.0f, 3.0f, 4.0f,
             5.0f, 6.0f, 7.0f, 8.0f,
             9.0f, 10.0f, 11.0f, 12.0f,
             13.0f, 14.0f, 15.0f, 16.0f};
    let i = identity4();
    let c = matmul4(&a, &i);

    println("c[0] = {c[0]}  c[5] = {c[5]}");
    println("c[10] = {c[10]}  c[15] = {c[15]}");
}
