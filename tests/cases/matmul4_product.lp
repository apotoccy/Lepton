import "std/nn/matmul.lp";

fn main() {
    let a = {1.0f, 2.0f, 0.0f, 0.0f,
             0.0f, 1.0f, 3.0f, 0.0f,
             0.0f, 0.0f, 1.0f, 4.0f,
             5.0f, 0.0f, 0.0f, 1.0f};
    let b = {1.0f, 0.0f, 0.0f, 0.0f,
             2.0f, 1.0f, 0.0f, 0.0f,
             0.0f, 3.0f, 1.0f, 0.0f,
             0.0f, 0.0f, 4.0f, 1.0f};
    let c = matmul4(&a, &b);

    println("c = {c[0]} {c[1]} {c[2]} {c[3]}");
    println("c = {c[4]} {c[5]} {c[6]} {c[7]}");
    println("c = {c[8]} {c[9]} {c[10]} {c[11]}");
    println("c = {c[12]} {c[13]} {c[14]} {c[15]}");
}
