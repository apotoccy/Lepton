struct Inner {
    x: i32;
}

struct P {
    x: i32;
    inner: Inner;
    arr: [i32; 3];
}

struct Item {
    value: i32;
}

fn main() {
    var p = P(x: 1, inner: Inner(x: 2), arr: {3, 4, 5});
    p.x = 5;
    p.inner.x = 6;
    p.arr[2] = 7;

    var a: [i32; 3] = {1, 2, 3};
    a[1] = 8;

    var v: v4i32 = [1, 2, 3, 4];
    v[2] = 9;

    let q = alloc([i32; 2]);
    q[0] = 10;

    var grid: [[i32; 2]; 2] = {{0, 0}, {0, 0}};
    grid[0][0] = 4;
    grid[1][1] += 5;

    var items: [Item; 2] = {Item(value: 0), Item(value: 0)};
    items[0] = Item(value: 1);
    items[0].value = 11;

    var i = 0;
    p.arr[i] += 10;
    a[0] += 4;
    v[1] += 6;
    q[0] += 1;

    println("p = {p.x} {p.inner.x} {p.arr[0]} {p.arr[1]} {p.arr[2]}");
    println("a = {a[0]} {a[1]}");
    println("v = {v[1]} {v[2]}");
    println("q = {q[0]}");
    println("grid = {grid[0][0]} {grid[1][1]}");
    println("item = {items[0].value}");
}
