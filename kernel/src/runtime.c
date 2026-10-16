/* Lepton kernel runtime: satisfies every lp_* symbol the compiler emits
   in --freestanding mode, and provides memset/memcpy for GCC. */

typedef unsigned int   uint32_t;
typedef unsigned char  uint8_t;
typedef __SIZE_TYPE__  size_t;

/* ---------- port I/O ---------- */
static inline void outb(uint32_t port, uint8_t v) {
    __asm__ volatile("outb %0, %1" : : "a"(v), "Nd"(port));
}
static inline uint8_t inb(uint32_t port) {
    uint8_t r;
    __asm__ volatile("inb %1, %0" : "=a"(r) : "Nd"(port));
    return r;
}

/* ---------- COM1 serial ---------- */
#define COM1 0x3F8
void serial_init(void) {
    outb(COM1 + 1, 0x00);
    outb(COM1 + 3, 0x80);
    outb(COM1 + 0, 0x03);
    outb(COM1 + 1, 0x00);
    outb(COM1 + 3, 0x03);
    outb(COM1 + 2, 0xC7);
    outb(COM1 + 4, 0x0B);
}
static int serial_ready(void) { return inb(COM1 + 5) & 0x20; }
static void serial_putc(char c) {
    while (!serial_ready()) { }
    outb(COM1, (uint8_t)c);
}

/* ---------- compiler runtime ---------- */
#define HEAP_SIZE (64 * 1024)
static unsigned char heap[HEAP_SIZE];
static unsigned long heap_used = 0;

char *lp_alloc(unsigned long n) {
    n = (n + 15UL) & ~15UL;
    if (heap_used + n > HEAP_SIZE) return (char *)0;
    char *p = (char *)&heap[heap_used];
    heap_used += n;
    return p;
}

void lp_print(const char *s)  { while (*s) serial_putc(*s++); }
void lp_println(const char *s) { lp_print(s); serial_putc('\r'); serial_putc('\n'); }

void lp_panic(const char *s) {
    lp_print("PANIC: ");
    lp_println(s);
    for (;;) __asm__ volatile("cli; hlt");
}

const char *lp_cat(const char *a, const char *b) {
    unsigned long la = 0, lb = 0, i;
    while (a[la]) la++;
    while (b[lb]) lb++;
    char *r = lp_alloc(la + lb + 1);
    if (!r) return a;
    for (i = 0; i < la; i++) r[i] = a[i];
    for (i = 0; i < lb; i++) r[la + i] = b[i];
    r[la + lb] = 0;
    return r;
}

const char *lp_str_i64(long long v) {
    static char bufs[4][24];
    static int  idx = 0;
    char *r = bufs[idx]; idx = (idx + 1) & 3;

    int neg = v < 0;
    unsigned long long u = neg ? (unsigned long long)(-(v + 1)) + 1ULL
                               : (unsigned long long)v;
    int i = 0;
    do { r[i++] = (char)('0' + (int)(u % 10ULL)); u /= 10ULL; } while (u);
    if (neg) r[i++] = '-';
    r[i] = 0;
    for (int j = 0; j < i / 2; j++) {
        char t = r[j]; r[j] = r[i - 1 - j]; r[i - 1 - j] = t;
    }
    return r;
}

const char *lp_str_u64(unsigned long long u) {
    static char bufs[4][24];
    static int  idx = 0;
    char *r = bufs[idx]; idx = (idx + 1) & 3;
    int i = 0;
    do { r[i++] = (char)('0' + (int)(u % 10ULL)); u /= 10ULL; } while (u);
    r[i] = 0;
    for (int j = 0; j < i / 2; j++) {
        char t = r[j]; r[j] = r[i - 1 - j]; r[i - 1 - j] = t;
    }
    return r;
}

const char *lp_str_bool(int v) { return v ? "true" : "false"; }

const char *lp_str_char(char c) {
    static char buf[2];
    buf[0] = c; buf[1] = 0;
    return buf;
}

const char *lp_str_f64(double v) {
    /* Floats are rarely useful in a kernel; provide a placeholder. */
    (void)v;
    return "<f64>";
}

/* ---------- GCC builtins ---------- */
void *memset(void *d, int c, size_t n) {
    unsigned char *p = (unsigned char *)d;
    while (n--) *p++ = (unsigned char)c;
    return d;
}
void *memcpy(void *d, const void *s, size_t n) {
    unsigned char *dp = (unsigned char *)d;
    const unsigned char *sp = (const unsigned char *)s;
    while (n--) *dp++ = *sp++;
    return d;
}
void *memmove(void *d, const void *s, size_t n) {
    unsigned char *dp = (unsigned char *)d;
    const unsigned char *sp = (const unsigned char *)s;
    if (dp < sp) { while (n--) *dp++ = *sp++; }
    else         { dp += n; sp += n; while (n--) *--dp = *--sp; }
    return d;
}
int memcmp(const void *a, const void *b, size_t n) {
    const unsigned char *x = a, *y = b;
    while (n--) { if (*x != *y) return (int)*x - (int)*y; x++; y++; }
    return 0;
}
