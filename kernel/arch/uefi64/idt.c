/* Minimal exception IDT; interrupts remain disabled in native mode. */
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

extern void isr_default(void);
void platform_puts(const char *s);

void native_exception(void) {
    platform_puts("\r\nEXCEPTION: native 64-bit exception; halted\r\n");
    for (;;) { __asm__ volatile("cli; hlt"); }
}

void native_idt_init(void) {
    uint16_t cs = 0;
    __asm__ volatile("mov %%cs, %0" : "=r"(cs));
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
    idt_ptr p = { (uint16_t)(sizeof(idt) - 1), (uint64_t)(uintptr_t)idt };
    __asm__ volatile("lidt %0" : : "m"(p));
}
