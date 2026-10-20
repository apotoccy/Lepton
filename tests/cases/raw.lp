struct Point {
    x: i32;
    y: i32;
}

fn main() {
    var x = 10;
    let p = &x;
    raw {
        println("{*p}");
        *p = 20;
        *p += 5;
    }
    println("{x}");

    var arr: [i32; 3] = {1, 2, 3};
    let ap = &arr;
    raw {
        println("{(*ap)[1]}");
        (*ap)[1] = 9;
        (*ap)[0] += 4;
    }
    println("{arr[0]} {arr[1]} {arr[2]}");

    var pt = Point(x: 1, y: 2);
    let pp = &pt;
    raw {
        (*pp).x = 7;
        (*pp).y += 3;
    }
    println("{pt.x} {pt.y}");
}
