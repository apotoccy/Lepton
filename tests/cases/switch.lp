enum Color { Red, Green, Blue }

const LIMIT: i32 = 1;

fn color_name(c: Color) -> str {
    switch (c) {
        case Color.Red { return "red"; }
        case Color.Green { return "green"; }
        default { return "blue"; }
    }
}

fn classify(x: i32) -> i32 {
    var out = 0;
    switch (x) {
        case 0, 1 { out = 10; }
        case 2 { out = 20; break; }
        case 3 { out = 30; }
        default { out = 99; }
    }
    return out;
}

fn main() {
    println("{classify(0)} {classify(1)} {classify(2)} {classify(3)} {classify(4)}");
    println("{color_name(Color.Red)} {color_name(Color.Green)} {color_name(Color.Blue)}");

    switch ('a') {
        case 'a' { println("char"); }
        default { println("other"); }
    }

    switch (1) {
        case LIMIT { println("const {LIMIT}"); }
        default { println("other"); }
    }

    var i = 0;
    var hits = 0;
    while (i < 4) {
        switch (i) {
            case 0 { hits += 1; i += 1; continue; }
            case 1 { hits += 10; break; }
            default { hits += 100; }
        }
        i += 1;
    }
    println("{i} {hits}");
}
