/* Interrupt subsystem: GDT, IDT, PIC remap, PIT timer. */

typedef unsigned char  u8;
typedef unsigned short u16;
typedef unsigned int   u32;

extern void vga_putc(char c);

static inline void outb(u16 port, u8 v) {
    __asm__ volatile("outb %0, %1" : : "a"(v), "Nd"(port));
}
static inline u8 inb(u16 port) {
    u8 r;
    __asm__ volatile("inb %1, %0" : "=a"(r) : "Nd"(port));
    return r;
}
static inline void io_wait(void) { outb(0x80, 0); }

/* ---- Serial output (used by panic and debug) ---- */
static inline void serial_putc(char c) {
    while ((inb(0x3F8 + 5) & 0x20) == 0) { }
    outb(0x3F8, (u8)c);
}
static void serial_puts(const char *s) {
    while (*s) serial_putc(*s++);
}
static void serial_hex32(u32 v) {
    static const char h[] = "0123456789ABCDEF";
    for (int i = 28; i >= 0; i -= 4) serial_putc(h[(v >> i) & 0xF]);
}

/* ---- GDT ---- */
#include <stdint.h>

static uint64_t gdt_entries[3];

static uint64_t gdt_entries[3];
struct __attribute__((packed)) gdt_ptr_t {
    uint16_t limit;
    uint32_t base;
};
static struct gdt_ptr_t gdt_ptr;

extern void gdt_load(struct gdt_ptr_t *);

static void gdt_init(void) {
    /* Null descriptor */
    gdt_entries[0] = 0;

    /* Code: base 0, limit 0xFFFFF, ring 0, 4 KiB gran, 32-bit, exec/read.
       Access byte 0x9A = present, ring 0, code, readable.
       Flags byte 0xCF = granularity 4K, 32-bit, limit high nibble F. */
    gdt_entries[1] = 0x00CF9A000000FFFFULL;

    /* Data: same but writable. Access 0x92 = present, ring 0, data, writable. */
    gdt_entries[2] = 0x00CF92000000FFFFULL;

    gdt_ptr.limit = sizeof(gdt_entries) - 1;
    gdt_ptr.base  = (uint32_t)gdt_entries;
    gdt_load(&gdt_ptr);
}

/* ---- IDT ---- */
struct __attribute__((packed)) idt_entry_t {
    uint16_t offset_low;
    uint16_t selector;
    uint8_t  zero;
    uint8_t  type_attr;
    uint16_t offset_high;
};

struct __attribute__((packed)) idt_ptr_t {
    uint16_t limit;
    uint32_t base;
};

static struct idt_entry_t idt[256];
static struct idt_ptr_t   idt_ptr;

extern void isr0(void);  extern void isr1(void);  extern void isr2(void);
extern void isr3(void);  extern void isr4(void);  extern void isr5(void);
extern void isr6(void);  extern void isr7(void);  extern void isr8(void);
extern void isr9(void);  extern void isr10(void); extern void isr11(void);
extern void isr12(void); extern void isr13(void); extern void isr14(void);
extern void isr15(void); extern void isr16(void); extern void isr17(void);
extern void isr18(void); extern void isr19(void); extern void isr20(void);
extern void isr21(void); extern void isr22(void); extern void isr23(void);
extern void isr24(void); extern void isr25(void); extern void isr26(void);
extern void isr27(void); extern void isr28(void); extern void isr29(void);
extern void isr30(void); extern void isr31(void);

extern void isr32(void); extern void isr33(void); extern void isr34(void);
extern void isr35(void); extern void isr36(void); extern void isr37(void);
extern void isr38(void); extern void isr39(void); extern void isr40(void);
extern void isr41(void); extern void isr42(void); extern void isr43(void);
extern void isr44(void); extern void isr45(void); extern void isr46(void);
extern void isr47(void);

static void idt_set(int n, void (*handler)(void)) {
    uint32_t addr = (uint32_t)handler;
    idt[n].offset_low  = addr & 0xFFFF;
    idt[n].selector    = 0x08;
    idt[n].zero        = 0;
    idt[n].type_attr   = 0x8E;   /* present, ring 0, 32-bit interrupt gate */
    idt[n].offset_high = (addr >> 16) & 0xFFFF;
}

static void idt_init(void) {
    void (*handlers[48])(void) = {
        isr0, isr1, isr2, isr3, isr4, isr5, isr6, isr7,
        isr8, isr9, isr10, isr11, isr12, isr13, isr14, isr15,
        isr16, isr17, isr18, isr19, isr20, isr21, isr22, isr23,
        isr24, isr25, isr26, isr27, isr28, isr29, isr30, isr31,
        isr32, isr33, isr34, isr35, isr36, isr37, isr38, isr39,
        isr40, isr41, isr42, isr43, isr44, isr45, isr46, isr47
    };
    for (int i = 0; i < 48; i++) idt_set(i, handlers[i]);
    /* 48..255 stay zeroed — any spurious vector hits a null descriptor
       and the CPU raises #GP, which reaches isr13. */
    idt_ptr.limit = sizeof(idt) - 1;
    idt_ptr.base  = (uint32_t)idt;
    __asm__ volatile("lidtl %0" : : "m"(idt_ptr));
}

/* ---- PIC ---- */
#define PIC1_CMD  0x20
#define PIC1_DATA 0x21
#define PIC2_CMD  0xA0
#define PIC2_DATA 0xA1

static void pic_remap(void) {
    u8 mask1 = inb(PIC1_DATA);
    u8 mask2 = inb(PIC2_DATA);

    outb(PIC1_CMD, 0x11); io_wait();    /* ICW1: init, expect ICW4 */
    outb(PIC2_CMD, 0x11); io_wait();
    outb(PIC1_DATA, 0x20); io_wait();   /* ICW2: IRQ 0..7 → 0x20..0x27 */
    outb(PIC2_DATA, 0x28); io_wait();   /* ICW2: IRQ 8..15 → 0x28..0x2F */
    outb(PIC1_DATA, 0x04); io_wait();   /* ICW3: slave on IRQ 2 */
    outb(PIC2_DATA, 0x02); io_wait();   /* ICW3: slave cascade identity */
    outb(PIC1_DATA, 0x01); io_wait();   /* ICW4: 8086 mode */
    outb(PIC2_DATA, 0x01); io_wait();

    /* Unmask IRQ 0 (timer) and IRQ 1 (keyboard). */
    outb(PIC1_DATA, 0xEC);              /* 1110_1100 = IRQ 0, 1, 4 enabled */
    outb(PIC2_DATA, 0xFF);              /* all slave IRQs masked */

    (void)mask1; (void)mask2;
}

static void pic_eoi(int irq) {
    if (irq >= 8) outb(PIC2_CMD, 0x20);
    outb(PIC1_CMD, 0x20);
}

/* ---- PIT (100 Hz) ---- */
static void pit_init(void) {
    u32 divisor = 1193182 / 100;        /* PIT frequency / desired Hz */
    outb(0x43, 0x36);                   /* channel 0, lo/hi, mode 3 (square) */
    outb(0x40, (u8)(divisor & 0xFF));
    outb(0x40, (u8)((divisor >> 8) & 0xFF));
}

/* ---- Dispatcher ---- */
static volatile uint32_t tick_count = 0;

/* ---- Keyboard ---- */
static u8 shift_down = 0;
static u8 ctrl_down = 0;
static u8 alt_down = 0;

extern void lp_reboot(void);

/* Scancode set 1 (make codes). Index is the raw scancode 0x00..0x7F.
   Zero entries are keys we don't handle yet (F-keys, arrows, ctrl, alt). */
static const char kbd_normal[128] = {
    0,    27,   '1',  '2',  '3',  '4',  '5',  '6',
    '7',  '8',  '9',  '0',  '-',  '=',  '\b', '\t',
    'q',  'w',  'e',  'r',  't',  'y',  'u',  'i',
    'o',  'p',  '[',  ']',  '\n',  0,   'a',  's',
    'd',  'f',  'g',  'h',  'j',  'k',  'l',  ';',
    '\'', '`',  0,    '\\', 'z',  'x',  'c',  'v',
    'b',  'n',  'm',  ',',  '.',  '/',  0,    '*',
    0,    ' ',  0,    0,    0,    0,    0,    0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

static const char kbd_shift[128] = {
    0,    27,   '!',  '@',  '#',  '$',  '%',  '^',
    '&',  '*',  '(',  ')',  '_',  '+',  '\b', '\t',
    'Q',  'W',  'E',  'R',  'T',  'Y',  'U',  'I',
    'O',  'P',  '{',  '}',  '\n',  0,   'A',  'S',
    'D',  'F',  'G',  'H',  'J',  'K',  'L',  ':',
    '"',  '~',  0,    '|',  'Z',  'X',  'C',  'V',
    'B',  'N',  'M',  '<',  '>',  '?',  0,    '*',
    0,    ' ',  0,    0,    0,    0,    0,    0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

#define KBD_BUF_SIZE 256
#define KBD_UP      0x100u
#define KBD_DOWN    0x101u
#define KBD_LEFT    0x102u
#define KBD_RIGHT   0x103u
#define KBD_HOME    0x104u
#define KBD_END     0x105u
#define KBD_DELETE  0x106u
#define KBD_TAB     0x107u
#define KBD_CTRL_C  0x108u
#define KBD_CTRL_L  0x109u
#define KBD_PGUP    0x10Au
#define KBD_PGDN    0x10Bu

static u32 kbd_buf[KBD_BUF_SIZE];
static volatile u16 kbd_head = 0;
static volatile u16 kbd_tail = 0;

static void kbd_push(u32 ev) {
    u16 next = (kbd_head + 1) % KBD_BUF_SIZE;
    if (next != kbd_tail) {
        kbd_buf[kbd_head] = ev;
        kbd_head = next;
    }
}

/* The REPL owns echoing and line editing; the driver only queues events. */
static void kbd_emit(char c) {
    kbd_push((u32)(uint8_t)c);
}

static u8 kbd_extended = 0;
static volatile u8 kbd_irq_seen = 0;

static void kbd_handle(u8 sc) {
    if (sc == 0xE0) { kbd_extended = 1; return; }
    if (kbd_extended) {
        kbd_extended = 0;
        if (sc == 0x48) { kbd_push(KBD_UP); }
        else if (sc == 0x50) { kbd_push(KBD_DOWN); }
        else if (sc == 0x4B) { kbd_push(KBD_LEFT); }
        else if (sc == 0x4D) { kbd_push(KBD_RIGHT); }
        else if (sc == 0x47) { kbd_push(KBD_HOME); }
        else if (sc == 0x4F) { kbd_push(KBD_END); }
        else if (sc == 0x53) { kbd_push(KBD_DELETE); }
        else if (sc == 0x49) { kbd_push(KBD_PGUP); }
        else if (sc == 0x51) { kbd_push(KBD_PGDN); }
        return;
    }
    if (sc == 0x2A || sc == 0x36) { shift_down = 1; return; }
    if (sc == 0xAA || sc == 0xB6) { shift_down = 0; return; }
    if (sc == 0x1D) { ctrl_down = 1; return; }
    if (sc == 0x9D) { ctrl_down = 0; return; }
    if (sc == 0x38) { alt_down = 1; return; }
    if (sc == 0xB8) { alt_down = 0; return; }
    if (sc == 0x53 && ctrl_down && alt_down) { lp_reboot(); }
    if (ctrl_down && sc == 0x2E) { kbd_push(KBD_CTRL_C); return; }
    if (ctrl_down && sc == 0x26) { kbd_push(KBD_CTRL_L); return; }
    if (ctrl_down && sc == 0x19) { kbd_push(KBD_PGUP); return; }
    if (ctrl_down && sc == 0x31) { kbd_push(KBD_PGDN); return; }
    if (sc == 0x0F) { kbd_push(KBD_TAB); return; }
    if (sc & 0x80) return;
    char c = shift_down ? kbd_shift[sc] : kbd_normal[sc];
    if (c == 0) return;
    kbd_emit(c);
}

void kbd_poll(void) {
    if (kbd_irq_seen) return;
    u8 status = inb(0x64);
    if ((status & 0x01) != 0 && (status & 0x20) == 0) {
        kbd_handle(inb(0x60));
    }
}

uint32_t kbd_read(void) {
    if (kbd_tail == kbd_head) return 0;
    u32 ev = kbd_buf[kbd_tail];
    kbd_tail = (kbd_tail + 1) % KBD_BUF_SIZE;
    return ev;
}

uint32_t kbd_cancel_pending(void) {
    u16 i = kbd_tail;
    while (i != kbd_head) {
        if (kbd_buf[i] == KBD_CTRL_C || kbd_buf[i] == 3u) return 1;
        i = (i + 1) % KBD_BUF_SIZE;
    }
    return 0;
}

void irq_handler(uint32_t *frame) {
    uint32_t vec = frame[8];            /* [8] is int#, [9] is err */
    (void)frame;

    if (vec < 32) {
        /* CPU exception! Print to VGA so we can see it on the laptop */
        const char *msg = "\nEXCEPTION ";
        while (*msg) vga_putc(*msg++);

        /* Print the exception number in hex */
        static const char hex[] = "0123456789ABCDEF";
        vga_putc(hex[(vec >> 4) & 0xF]);
        vga_putc(hex[vec & 0xF]);
        vga_putc('\n');

        /* CPU exception. Without a proper handler, halt with a message. */
//        serial_puts("\nEXCEPTION ");
//        serial_hex32(vec);
//        serial_puts(" - halting\n");

        for (;;) __asm__ volatile("cli; hlt");
    }

    int irq = (int)vec - 32;
    if (irq == 0) {
        tick_count++;
    }
    /* IRQ 1 (keyboard) — read and discard until a real driver lands. */
    if (irq == 1) {
        kbd_irq_seen = 1;
        kbd_handle(inb(0x60));
    }
    if (irq == 4) {
        while ((inb(0x3F8 + 5) & 0x01) != 0) {
            kbd_emit((char)inb(0x3F8));
        }
    }

    pic_eoi(irq);
}


/* ---- PS/2 keyboard controller ---- */
static int kbd_wait_input(void) {
    for (int i = 0; i < 100000; i++) {
        if ((inb(0x64) & 0x02) == 0) return 1;
    }
    return 0;
}

static int kbd_wait_output(void) {
    for (int i = 0; i < 100000; i++) {
        if ((inb(0x64) & 0x01) != 0) return 1;
    }
    return 0;
}

static void kbd_send(u8 v) {
    if (!kbd_wait_input()) return;
    outb(0x60, v);
    if (kbd_wait_output()) (void)inb(0x60);
}

static void keyboard_init(void) {
    /* Disable both devices and flush a possibly stale output buffer. */
    if (kbd_wait_input()) outb(0x64, 0xAD);
    if (kbd_wait_input()) outb(0x64, 0xA7);
    for (int i = 0; i < 32 && (inb(0x64) & 0x01) != 0; i++) {
        (void)inb(0x60);
    }

    /* Read the controller configuration byte, enable IRQ 1, enable the
       keyboard clock, and request scancode translation to set 1. */
    if (kbd_wait_input()) outb(0x64, 0x20);
    if (kbd_wait_output()) {
        u8 cfg = inb(0x60);
        cfg |= 0x01;      /* IRQ 1 */
        cfg &= (u8)~0x10; /* keyboard clock enabled */
        cfg |= 0x40;      /* translate to scancode set 1 */
        if (kbd_wait_input()) outb(0x64, 0x60);
        if (kbd_wait_input()) outb(0x60, cfg);
    }

    /* Enable the keyboard, select scancode set 2, and start scanning.
       The controller translation bit above turns set 2 into the set-1
       codes used by the key tables. */
    if (kbd_wait_input()) outb(0x64, 0xAE);
    kbd_send(0xF0);
    kbd_send(0x02);
    kbd_send(0xF4);
}

/* ---- Public API ---- */
void interrupts_init(void) {
    gdt_init();
    idt_init();
    keyboard_init();
    pic_remap();
    pit_init();
    /* Enable COM1 receive-data interrupt. serial_init already set the
       UART to 8N1, so we only need IER bit 0. */
    outb(0x3F8 + 1, 0x01);
    __asm__ volatile("sti");
}

uint32_t ticks(void) {
    return tick_count;
}

void cpu_hlt(void) {
    __asm__ volatile("hlt");
}
