/* Lepton kernel runtime: satisfies every lp_* symbol the compiler emits
   in --freestanding mode, and provides memset/memcpy for GCC. */

extern void vga_putc(char c);
extern void vga_clear(void);

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

void lp_alloc_init(void *start, unsigned long size) {
    heap_start = (unsigned char *)start;
    heap_used = 0;
    heap_max = size;
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

__attribute__((noreturn)) void lp_panic(const char *s) {
    lp_print("PANIC: ");
    lp_println(s);
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
