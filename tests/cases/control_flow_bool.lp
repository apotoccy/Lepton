fn main() {
    let a = true;
    let b = false;

    if (!b && a) {
        println("and/not ok");
    }
    if (a == true) {
        println("eq ok");
    }
    if (a != b) {
        println("ne ok");
    }

    var i = 0;
    while (true) {
        i = i + 1;
        if (i == 3) {
            break;
        }
    }
    println("loop = {i}");
}
