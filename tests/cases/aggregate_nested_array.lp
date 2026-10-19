struct Grid {
    cells: [[f32; 2]; 2];
}

fn main() {
    let g = Grid(cells: {{1.0f, 2.0f}, {3.0f, 4.0f}});
    println("{g.cells[0][0]} {g.cells[1][1]}");
}
