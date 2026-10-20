fn main() {
    var x: i32;
    x = 5;
    if (x > 0) {
        x = 6;
    }
    println("{x}");

    var y: i32;
    if (x > 0) {
        y = 1;
    } else {
        y = 2;
    }
    println("{y}");

    var z: i32;
    switch (x) {
        case 6 { z = 1; }
        default { z = 2; }
    }
    println("{z}");

    var a: [i32; 2] = {0, 0};
    a[0] = 1;
    println("{a[0]}");
}
