import "std/nn/matmul.lp";

fn main() {
    var a = {1.0f, 2.0f, 3.0f, 4.0f,
             5.0f, 6.0f, 7.0f, 8.0f,
             9.0f, 10.0f, 11.0f, 12.0f,
             13.0f, 14.0f, 15.0f, 16.0f};
    let b = {1.0f, 0.0f, 0.0f, 0.0f,
             0.0f, 1.0f, 0.0f, 0.0f,
             0.0f, 0.0f, 1.0f, 0.0f,
             0.0f, 0.0f, 0.0f, 1.0f};

    add_inplace(&a, &b);

    println("a[0] = {a[0]}  a[5] = {a[5]}");
    println("a[10] = {a[10]}  a[15] = {a[15]}");
}
