struct Node {
    next: ptr[Node];
    value: i32;
}

fn by_value(a: [i32; 4]) -> void {
    a[0] = 99;
}

fn by_ref(a: ptr[[i32; 4]]) -> void {
    a[0] = 99;
}

fn main() {
    let n = Node(next: 0, value: 7);
    println("node = {n.value}");

    var a: [i32; 4];
    a[0] = 7;
    by_value(a);
    println("value = {a[0]}");
    by_ref(&a);
    println("ref = {a[0]}");
}
