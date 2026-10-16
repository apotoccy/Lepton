fn main() {
    let x = 0.5 as q7;
    let y = x as f32;
    let z = x as q15;

    println("x = {x}");
    println("y = {y}");
    println("z = {z}");

    let big = 0.99 as q7;
    let over = big + big;

    println("big = {big}");
    println("over = {over}");
}
