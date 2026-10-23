/* Minimal exception IDT and legacy PIC/PIT timer for native UEFI mode. */
#include <stdint.h>

typedef struct {
    uint16_t off_lo;
    uint16_t selector;
    uint8_t  ist;
    uint8_t  type;
    uint16_t off_mid;
    uint32_t off_hi;
    uint32_t zero;
} __attribute__((packed)) idt_entry;

typedef struct {
    uint16_t limit;
    uint64_t base;
} __attribute__((packed)) idt_ptr;

static idt_entry idt[256];
static uint16_t idt_cs;
volatile uint64_t native_tick;

extern void isr_default(void);
extern void irq0_stub(void);
void platform_puts(const char *s);

static inline void outb(uint16_t port, uint8_t v) {
    __asm__ volatile("outb %0, %1" : : "a"(v), "Nd"(port));
}
static inline void io_wait(void) { outb(0x80, 0); }

void native_exception(void) {
    platform_puts("\r\nEXCEPTION: native 64-bit exception; halted\r\n");
    for (;;) { __asm__ volatile("cli; hlt"); }
}

static void set_gate(int vec, void *handler) {
    uint64_t base = (uint64_t)(uintptr_t)handler;
    idt[vec].off_lo = (uint16_t)(base & 0xFFFF);
    idt[vec].selector = idt_cs;
    idt[vec].ist = 0;
    idt[vec].type = 0x8E;
    idt[vec].off_mid = (uint16_t)((base >> 16) & 0xFFFF);
    idt[vec].off_hi = (uint32_t)(base >> 32);
    idt[vec].zero = 0;
}

void native_idt_init(void) {
    uint16_t cs = 0;
    __asm__ volatile("mov %%cs, %0" : "=r"(cs));
    idt_cs = cs;
    uint64_t base = (uint64_t)(uintptr_t)isr_default;
    for (int i = 0; i < 256; i++) {
        idt[i].off_lo = (uint16_t)(base & 0xFFFF);
        idt[i].selector = cs;
        idt[i].ist = 0;
        idt[i].type = 0x8E;
        idt[i].off_mid = (uint16_t)((base >> 16) & 0xFFFF);
        idt[i].off_hi = (uint32_t)(base >> 32);
        idt[i].zero = 0;
    }
    set_gate(32, (void *)irq0_stub);
    idt_ptr p = { (uint16_t)(sizeof(idt) - 1), (uint64_t)(uintptr_t)idt };
    __asm__ volatile("lidt %0" : : "m"(p));
}

void native_irq_init(void) {
    /* Remap PIC to vectors 32..47. */
    outb(0x20, 0x11); io_wait();
    outb(0xA0, 0x11); io_wait();
    outb(0x21, 0x20); io_wait();
    outb(0xA1, 0x28); io_wait();
    outb(0x21, 0x04); io_wait();
    outb(0xA1, 0x02); io_wait();
    outb(0x21, 0x01); io_wait();
    outb(0xA1, 0x01); io_wait();
    /* Unmask only IRQ0; keyboard stays on polling. */
    outb(0x21, 0xFE);
    outb(0xA1, 0xFF);
    /* PIT channel 0, 100 Hz. */
    uint16_t div = (uint16_t)(1193182 / 100);
    outb(0x43, 0x36);
    outb(0x40, (uint8_t)(div & 0xFF));
    outb(0x40, (uint8_t)(div >> 8));
    __asm__ volatile("sti");
}
