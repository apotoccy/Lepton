/* Lepton kernel runtime: satisfies every lp_* symbol the compiler emits
   in --freestanding mode, and provides memset/memcpy for GCC. */

extern void vga_putc(char c);
extern void vga_clear(void);
extern void vga_cursor_left(void);
extern void vga_cursor_right(void);

typedef unsigned int   uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char  uint8_t;
typedef __SIZE_TYPE__  size_t;

/* ---------- port I/O ---------- */
static inline void outb(uint16_t port, uint8_t v) {
    __asm__ volatile("outb %0, %1" : : "a"(v), "Nd"(port));
}
static inline uint8_t inb(uint16_t port) {
    uint8_t r;
    __asm__ volatile("inb %1, %0" : "=a"(r) : "Nd"(port));
    return r;
}
static inline void outw(uint16_t port, uint16_t v) {
    __asm__ volatile("outw %0, %1" : : "a"(v), "Nd"(port));
}
static inline uint16_t inw(uint16_t port) {
    uint16_t r;
    __asm__ volatile("inw %1, %0" : "=a"(r) : "Nd"(port));
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
//#define HEAP_SIZE (64 * 1024)
//static unsigned char heap[HEAP_SIZE];
/* 32 MB heap. Enough for a dequantized 768x768 f32 tensor (2.3 MB),
   plus scratch and model buffers. QEMU's default 128 MB RAM handles
   this comfortably. On real hardware with less than 64 MB, drop this
   to 8 MB and use smaller tensors. */
#define HEAP_SIZE (8 * 1024 * 1024)
static unsigned char heap[HEAP_SIZE];
static unsigned long heap_used = 0;

static unsigned char *heap_start = 0;
static unsigned long heap_max = 0;

void lp_alloc_init(void *start, uint32_t size) {
    heap_start = (unsigned char *)start;
    heap_used = 0;
    heap_max = (unsigned long)size;
}

char *lp_alloc(unsigned long n);

char *lp_alloc_bytes(long long n) {
    if (n <= 0) return (char *)0;
    return lp_alloc((unsigned long)n);
}

char *lp_alloc(unsigned long n) {
    n = (n + 15UL) & ~15UL;
    if (heap_start == 0) return (char *)0;
    if (heap_used + n > heap_max) return (char *)0;
    char *p = (char *)&heap_start[heap_used];
    heap_used += n;
    return p;
}

/* The bump allocator cannot reclaim one block. free() is a no-op here;
   alloc_reset() reclaims the whole heap. */
void lp_free(void *p) {
    (void)p;
}

void lp_alloc_reset(void) {
    heap_used = 0;
}

long long lp_heap_used(void) {
    return (long long)heap_used;
}

long long lp_heap_max(void) {
    return (long long)heap_max;
}

long long lp_alloc_mark(void) {
    return (long long)heap_used;
}

void lp_alloc_release(long long mark) {
    if (mark >= 0 && (unsigned long)mark <= heap_used) {
        heap_used = (unsigned long)mark;
    }
}

//char *lp_alloc(unsigned long n) {
//    n = (n + 15UL) & ~15UL;
//    if (heap_used + n > HEAP_SIZE) return (char *)0;
//    char *p = (char *)&heap[heap_used];
//    heap_used += n;
//    return p;
//}

void lp_print(const char *s) {
    while (*s) {
        char c = *s++;
        if (c == '\n') {
            serial_putc('\r');  /* serial needs CRLF */
            serial_putc('\n');
            vga_putc('\n');
        } else {
            serial_putc(c);
            vga_putc(c);
        }
    }
}

void lp_println(const char *s) {
    lp_print(s);
    serial_putc('\r');
    serial_putc('\n');
    vga_putc('\n');
}

void lp_cursor_left(void) {
    serial_putc(27);
    serial_putc('[');
    serial_putc('D');
    vga_cursor_left();
}

void lp_cursor_right(void) {
    serial_putc(27);
    serial_putc('[');
    serial_putc('C');
    vga_cursor_right();
}

void lp_clear_screen(void) {
    serial_putc(27);
    serial_putc('[');
    serial_putc('2');
    serial_putc('J');
    serial_putc(27);
    serial_putc('[');
    serial_putc('H');
    vga_clear();
}

__attribute__((noreturn)) void lp_panic(const char *s) {
    lp_print("PANIC: ");
    lp_println(s);
    lp_println("Ctrl+Alt+Del to reboot");
    for (;;) __asm__ volatile("sti; hlt");
}

__attribute__((noreturn)) void lp_reboot(void) {
    lp_println("rebooting...");
    outb(0x64, 0xFE);
    for (;;) __asm__ volatile("cli; hlt");
}

/* ---- ACPI power off ---------------------------------------------------- */
static int acpi_sig(const unsigned char *p, const char *sig) {
    for (int i = 0; i < 4; i++) {
        if (p[i] != (unsigned char)sig[i]) return 0;
    }
    return 1;
}

static unsigned char *acpi_find_rsdp(void) {
    unsigned int ebda = (unsigned int)(*(unsigned short *)0x040E) << 4;
    unsigned int ranges[2][2] = {
        { ebda, ebda + 1024 },
        { 0x000E0000, 0x000FFFFF },
    };
    for (int r = 0; r < 2; r++) {
        unsigned int start = ranges[r][0];
        unsigned int end = ranges[r][1];
        if (start == 0 || end <= start) continue;
        for (unsigned int a = start; a + 20 <= end; a += 16) {
            unsigned char *p = (unsigned char *)a;
            if (acpi_sig(p, "RSD ") && acpi_sig(p + 4, "PTR ")) {
                unsigned char sum = 0;
                for (int i = 0; i < 20; i++) sum = (unsigned char)(sum + p[i]);
                if (sum == 0) return p;
            }
        }
    }
    return (unsigned char *)0;
}

static unsigned char *acpi_find_fadt(void) {
    unsigned char *rsdp = acpi_find_rsdp();
    if (!rsdp) return (unsigned char *)0;
    unsigned int rsdt = *(unsigned int *)(rsdp + 16);
    unsigned char *p = (unsigned char *)rsdt;
    if (!acpi_sig(p, "RSDT")) return (unsigned char *)0;
    unsigned int len = *(unsigned int *)(p + 4);
    if (len < 36) return (unsigned char *)0;
    unsigned int n = (len - 36) / 4;
    for (unsigned int i = 0; i < n; i++) {
        unsigned int off = *(unsigned int *)(p + 36 + i * 4);
        unsigned char *t = (unsigned char *)off;
        if (acpi_sig(t, "FACP")) return t;
    }
    return (unsigned char *)0;
}

/* The SLP_TYP values live in the DSDT _S5 package, not in the FADT.
   This is a deliberately small AML scan for:
       Name (_S5, Package (N) { <a>, <b>, ... })
   It accepts the common integer encodings and ignores anything else. */
static int acpi_aml_int(const unsigned char *p, unsigned int len,
                        unsigned int *at, unsigned int *out) {
    if (*at >= len) return 0;
    unsigned char op = p[(*at)++];
    unsigned int v = 0;
    if (op == 0x00) {
        v = 0;
    } else if (op == 0x01) {
        v = 1;
    } else if (op == 0x0A) {
        if (*at + 1 > len) return 0;
        v = p[(*at)++];
    } else if (op == 0x0B) {
        if (*at + 2 > len) return 0;
        v = (unsigned int)p[*at] | ((unsigned int)p[*at + 1] << 8);
        *at += 2;
    } else if (op == 0x0C) {
        if (*at + 4 > len) return 0;
        v = (unsigned int)p[*at] | ((unsigned int)p[*at + 1] << 8)
          | ((unsigned int)p[*at + 2] << 16)
          | ((unsigned int)p[*at + 3] << 24);
        *at += 4;
    } else {
        return 0;
    }
    *out = v;
    return 1;
}

static int acpi_s5_types(unsigned char *dsdt, unsigned char *a,
                         unsigned char *b) {
    if (!dsdt || !acpi_sig(dsdt, "DSDT")) return 0;
    unsigned int len = *(unsigned int *)(dsdt + 4);
    if (len < 40) return 0;

    for (unsigned int p = 36; p + 6 < len; p++) {
        if (dsdt[p] != 0x08) continue;
        if (dsdt[p + 1] != '_' || dsdt[p + 2] != 'S'
                || dsdt[p + 3] != '5' || dsdt[p + 4] != '_') continue;
        unsigned int at = p + 5;
        if (at >= len || dsdt[at] != 0x12) continue;
        at++;
        if (at >= len) continue;
        unsigned char lead = dsdt[at++];
        unsigned int plen = 0;
        if ((lead & 0xC0) == 0) {
            plen = lead & 0x3F;
        } else {
            int n = lead >> 6;
            plen = lead & 0x0F;
            for (int i = 0; i < n; i++) {
                if (at >= len) return 0;
                plen |= (unsigned int)dsdt[at++] << (4 + 8 * i);
            }
        }
        unsigned int end = at + plen;
        if (end > len || at >= end) continue;
        unsigned int count = dsdt[at++];
        if (count == 0) continue;
        unsigned int va = 0, vb = 0;
        if (!acpi_aml_int(dsdt, end, &at, &va)) continue;
        if (count >= 2) {
            if (!acpi_aml_int(dsdt, end, &at, &vb)) continue;
        } else {
            vb = va;
        }
        if (va <= 7 && vb <= 7) {
            *a = (unsigned char)va;
            *b = (unsigned char)vb;
            return 1;
        }
    }
    return 0;
}

static unsigned int acpi_dsdt_addr(unsigned char *fadt, unsigned int len) {
    unsigned int dsdt = *(unsigned int *)(fadt + 40);
    if (dsdt == 0 && len >= 152) {
        unsigned long long x = *(unsigned long long *)(fadt + 144);
        if (x <= 0xFFFFFFFFULL) dsdt = (unsigned int)x;
    }
    return dsdt;
}

__attribute__((noreturn)) void lp_poweroff(void) {
    lp_println("powering off...");
    unsigned char *fadt = acpi_find_fadt();
    if (fadt) {
        unsigned int len = *(unsigned int *)(fadt + 4);
        if (len >= 90) {
            unsigned int pm1a = *(unsigned int *)(fadt + 64);
            unsigned int pm1b = *(unsigned int *)(fadt + 68);
            unsigned int smi = *(unsigned int *)(fadt + 48);
            unsigned char acpi_enable = fadt[52];
            unsigned char slp_a = 5, slp_b = 5;
            int have_s5 = 0;

            if (len >= 148) {
                unsigned int dsdt = acpi_dsdt_addr(fadt, len);
                have_s5 = acpi_s5_types((unsigned char *)dsdt, &slp_a, &slp_b);
            }
            if (!have_s5) {
                lp_println("ACPI: _S5 not found; assuming SLP_TYP 5");
            }

            if (pm1a != 0) {
                unsigned int val = inw((uint16_t)pm1a);
                if ((val & 1U) == 0 && smi != 0 && acpi_enable != 0) {
                    outb((uint16_t)smi, acpi_enable);
                    for (int i = 0; i < 100000; i++) {
                        if ((inw((uint16_t)pm1a) & 1U) != 0) break;
                    }
                    val = inw((uint16_t)pm1a);
                }
                outw((uint16_t)pm1a,
                     (uint16_t)((val & ~(7U << 10))
                                | ((unsigned int)slp_a << 10) | (1U << 13)));
            }
            if (pm1b != 0) {
                unsigned int val = inw((uint16_t)pm1b);
                outw((uint16_t)pm1b,
                     (uint16_t)((val & ~(7U << 10))
                                | ((unsigned int)slp_b << 10) | (1U << 13)));
            }

            /* Emulator fallback if the PM1 write did not take effect. */
            outw(0x604, 0x2000);
            outw(0xB004, 0x2000);
            outw(0x4004, 0x2000);
            lp_println("ACPI shutdown did not complete; halted");
            for (;;) __asm__ volatile("cli; hlt");
        }
    }

    /* QEMU/Bochs/VirtualBox ACPI shutdown ports. */
    outw(0x604, 0x2000);
    outw(0xB004, 0x2000);
    outw(0x4004, 0x2000);
    lp_println("poweroff is not supported by this firmware");
    for (;;) __asm__ volatile("cli; hlt");
}

/* There is no filesystem in the freestanding kernel. These entry points
   preserve the hosted API while returning a documented error. */
long long lp_file_size(const char *path) {
    (void)path;
    return -1;
}

long long lp_read_file_at(const char *path, long long offset,
                          void *buf, long long n) {
    (void)path; (void)offset; (void)buf; (void)n;
    return -1;
}

long long lp_write_file(const char *path, const void *buf, long long n) {
    (void)path; (void)buf; (void)n;
    return -1;
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

const char *lp_read_str(const void *buf, long long offset, long long n) {
    char *r = lp_alloc((unsigned long)(n + 1));
    if (!r) return "";
    const unsigned char *p = (const unsigned char *)buf + offset;
    for (long long i = 0; i < n; i++) r[i] = (char)p[i];
    r[n] = 0;
    return r;
}

/* Content-based string comparison. Null compares equal to null and sorts
   before any non-null string. */
int lp_str_eq(const char *a, const char *b) {
    if (a == b) return 1;
    if (!a || !b) return 0;
    while (*a && *b && *a == *b) { a++; b++; }
    return *a == *b;
}

int lp_str_cmp(const char *a, const char *b) {
    if (a == b) return 0;
    if (!a) return -1;
    if (!b) return 1;
    while (*a && *b && *a == *b) { a++; b++; }
    return (unsigned char)*a - (unsigned char)*b;
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

//const char *lp_str_char(char c) {
//    static char buf[2];
//    buf[0] = c; buf[1] = 0;
//    return buf;
//}

const char *lp_str_char(char c) {
    /* Four static buffers so `{a}{b}{c}{d}` renders four distinct chars. */
    static char bufs[4][2];
    static int idx = 0;
    char *b = bufs[idx];
    idx = (idx + 1) & 3;
    b[0] = c;
    b[1] = 0;
    return b;
}

const char *lp_str_f64(double v) {
    /* Four static buffers so `{a} {b} {c} {d}` prints four distinct values
       without aliasing. Sufficient for interpolation; not thread-safe. */
    static char bufs[4][40];
    static int idx = 0;
    char *buf = bufs[idx];
    idx = (idx + 1) & 3;
    int i = 0;

    /* NaN check: NaN != NaN */
    if (v != v) {
        buf[0]='n'; buf[1]='a'; buf[2]='n'; buf[3]=0;
        return buf;
    }

    /* Negative numbers */
    if (v < 0.0) {
        buf[i++] = '-';
        v = -v;
    }

    /* Very large: saturate to "inf" */
    if (v > 1e15) {
        buf[i++]='i'; buf[i++]='n'; buf[i++]='f'; buf[i]=0;
        return buf;
    }

    unsigned long long ip = (unsigned long long)v;
    double frac = v - (double)ip;

    /* Integer part, built in reverse */
    char tmp[24];
    int j = 0;
    if (ip == 0) {
        tmp[j++] = '0';
    } else {
        while (ip > 0) {
            tmp[j++] = (char)('0' + (int)(ip % 10));
            ip /= 10;
        }
    }
    while (j > 0) buf[i++] = tmp[--j];

    /* Fractional part: up to 6 digits, trailing zeros trimmed */
    if (frac > 1e-10) {
        buf[i++] = '.';
        int digits = 0;
        while (frac > 1e-10 && digits < 6) {
            frac *= 10.0;
            int d = (int)frac;
            buf[i++] = (char)('0' + d);
            frac -= (double)d;
            digits++;
        }
        while (i > 0 && buf[i-1] == '0') i--;
        if (i > 0 && buf[i-1] == '.') i--;
    }

    buf[i] = 0;
    return buf;
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
