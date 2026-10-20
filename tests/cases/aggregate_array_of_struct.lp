struct Point {
    x: i32;
    y: i32;
}

fn main() {
    var pts: [Point; 2] = {Point(x: 0, y: 0), Point(x: 0, y: 0)};
    pts[0] = Point(x: 1, y: 2);
    pts[1] = Point(x: 3, y: 4);
    println("{pts[0].x} {pts[0].y} {pts[1].x} {pts[1].y}");
}
