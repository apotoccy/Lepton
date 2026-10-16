extern fn serial_init() -> void;

fn main() {
    serial_init();
    println("Lepton OS kernel");
    println("booted on bare metal");

    for i in 0..4 {
        println("tick {i}");
    }

    println("halting.");
}
