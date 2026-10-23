/* UEFI64 platform shell: GOP text console, ConIn keyboard, heap, timer. */
#include "uefi.h"
#include "font8x16.h"

#ifdef UEFI_KERNEL
extern void main(void);
#endif

static EFI_SYSTEM_TABLE *ST;
static EFI_HANDLE g_image;
static EFI_GRAPHICS_OUTPUT_PROTOCOL *gop;
static UINT8 *fb;
static UINT32 fb_width, fb_height, fb_pitch, fb_format;
static UINT32 con_cols, con_rows, con_x, con_y;
static UINT32 fg_color = 0x00FFFFFF;
static UINT8 *heap_base;
static UINTN heap_size, heap_used;
static int mirror_serial = 1;
static UINTN map_count, map_pages;
static UINTN map_desc_size;

static inline void outb(UINT16 port, UINT8 v) {
    __asm__ volatile("outb %0, %1" : : "a"(v), "Nd"(port));
}
static inline UINT8 inb(UINT16 port) {
    UINT8 v;
    __asm__ volatile("inb %1, %0" : "=a"(v) : "Nd"(port));
    return v;
}
static void serial_init(void) {
    outb(0x3F8 + 1, 0x00); outb(0x3F8 + 3, 0x80);
    outb(0x3F8 + 0, 0x03); outb(0x3F8 + 1, 0x00);
    outb(0x3F8 + 3, 0x03); outb(0x3F8 + 2, 0xC7); outb(0x3F8 + 4, 0x0B);
}
static void serial_putc(char c) {
    while ((inb(0x3F8 + 5) & 0x20) == 0) { }
    outb(0x3F8, (UINT8)c);
}
static void serial_puts(const char *s) { while (*s) { serial_putc(*s++); } }
static void putpixel(UINT32 x, UINT32 y, UINT32 c) {
    if (x >= fb_width || y >= fb_height) return;
    *(UINT32 *)(fb + (UINTN)y * fb_pitch + (UINTN)x * 4) = c;
}
static void fill_rect(UINT32 x, UINT32 y, UINT32 w, UINT32 h, UINT32 c) {
    UINT32 yy, xx;
    for (yy = 0; yy < h; yy++) {
        for (xx = 0; xx < w; xx++) putpixel(x + xx, y + yy, c);
    }
}
static void draw_glyph(UINT32 cx, UINT32 cy, unsigned char ch) {
    const UINT8 *g;
    UINT32 row, col;
    if (ch < 32 || ch > 126) ch = '?';
    g = font8x16[ch - 32];
    for (row = 0; row < 16; row++) {
        UINT8 bits = g[row];
        for (col = 0; col < 8; col++) {
            putpixel(cx + col, cy + row, (bits & (0x80 >> col)) ? fg_color : 0);
        }
    }
}
static void con_clear(void) {
    fill_rect(0, 0, fb_width, fb_height, 0);
    con_x = 1; con_y = 1;
}
static void con_scroll(void) {
    UINTN row_bytes = (UINTN)fb_pitch * 16;
    UINTN total = (UINTN)fb_pitch * 16 * (con_rows - 1);
    UINT8 *tmp = fb;
    UINT8 *dst = fb;
    UINT8 *src = fb + row_bytes;
    UINTN i;
    for (i = 0; i < total; i++) dst[i] = src[i];
    fill_rect(0, (con_rows - 1) * 16, fb_width, 16, 0);
    (void)tmp;
}
static void con_putc(char c) {
    if (mirror_serial) serial_putc(c);
    if (c == '\n') { con_x = 0; con_y++; }
    else if (c == '\r') { con_x = 0; }
    else if (c == '\b') {
        if (con_x > 0) con_x--;
    } else {
        draw_glyph(con_x * 8, con_y * 16, (unsigned char)c);
        con_x++;
        if (con_x >= con_cols) { con_x = 0; con_y++; }
    }
    if (con_y >= con_rows) { con_scroll(); con_y = con_rows - 1; }
}
static void con_puts(const char *s) { while (*s) con_putc(*s++); }
static void con_dec(UINT64 v) {
    char b[24]; int n = 0;
    if (v == 0) { con_putc('0'); return; }
    while (v > 0 && n < 23) { b[n++] = (char)('0' + (v % 10)); v /= 10; }
    while (n > 0) con_putc(b[--n]);
}

static int str_eq(const char *a, const char *b) {
    while (*a && *b && *a == *b) { a++; b++; }
    return *a == 0 && *b == 0;
}
static int str_prefix(const char *a, const char *b) {
    while (*b) { if (*a++ != *b++) return 0; }
    return 1;
}

static void show_mem(void) {
    con_puts("heap: ");
    con_dec(heap_used);
    con_puts(" / ");
    con_dec(heap_size);
    con_puts(" bytes\r\n");
    con_puts("memory: descriptors ");
    con_dec(map_count);
    con_puts(", pages ");
    con_dec(map_pages);
    con_puts("\r\n");
}
static void show_config(void) {
    con_puts("platform: UEFI64\r\n");
    con_puts("firmware: ");
    if (ST->FirmwareVendor) {
        CHAR16 *v = ST->FirmwareVendor;
        while (*v) { con_putc((char)*v); v++; }
    }
    con_puts("\r\n");
    con_puts("mode: ");
    con_dec(fb_width); con_puts("x"); con_dec(fb_height);
    con_puts(" pitch "); con_dec(fb_pitch);
    con_puts("\r\n");
    con_puts("fb: "); con_dec(gop->Mode->FrameBufferBase);
    con_puts(" size "); con_dec(gop->Mode->FrameBufferSize);
    con_puts("\r\n");
}
static void show_time(void) {
    EFI_RUNTIME_SERVICES *rt = (EFI_RUNTIME_SERVICES *)ST->RuntimeServices;
    EFI_TIME t;
    if (!rt || !rt->GetTime || (rt->GetTime(&t, 0) & EFI_ERROR_MASK)) {
        con_puts("time unavailable\r\n");
        return;
    }
    con_puts("time: ");
    con_dec(t.Hour); con_puts(":");
    if (t.Minute < 10) con_puts("0");
    con_dec(t.Minute); con_puts(":");
    if (t.Second < 10) con_puts("0");
    con_dec(t.Second);
    con_puts(" UTC\r\n");
}

static void show_help(void) {
    con_puts("commands:\r\n");
    con_puts("  help       show this list\r\n");
    con_puts("  mem        heap and UEFI memory map summary\r\n");
    con_puts("  config     UEFI and framebuffer information\r\n");
    con_puts("  time       UEFI real-time clock\r\n");
    con_puts("  clear      clear the screen\r\n");
    con_puts("  alloc N    bump-allocate N bytes\r\n");
    con_puts("  echo TEXT  print text\r\n");
    con_puts("  reboot     cold reboot\r\n");
    con_puts("  poweroff   shut down\r\n");
    con_puts("  quit       return to firmware\r\n");
}

static EFI_STATUS heap_init(UINTN bytes) {
    void *p = 0;
    EFI_STATUS s = ST->BootServices->AllocatePool(4, bytes, &p);
    if (s & EFI_ERROR_MASK) return s;
    heap_base = (UINT8 *)p;
    heap_size = bytes;
    heap_used = 0;
    return 0;
}
static void *heap_alloc(UINTN bytes) {
    UINTN at = (heap_used + 15) & ~(UINTN)15;
    if (at + bytes > heap_size) return 0;
    heap_used = at + bytes;
    return heap_base + at;
}

#define INPUT_QUEUE 64
static UINT32 input_q[INPUT_QUEUE];
static UINT32 input_head, input_tail;

static UINT32 input_map(EFI_INPUT_KEY k) {
    if (k.UnicodeChar != 0) return (UINT32)k.UnicodeChar;
    if (k.ScanCode == 0x01) return 0x100;
    if (k.ScanCode == 0x02) return 0x101;
    if (k.ScanCode == 0x03) return 0x102;
    if (k.ScanCode == 0x04) return 0x103;
    if (k.ScanCode == 0x05) return 0x104;
    if (k.ScanCode == 0x06) return 0x105;
    if (k.ScanCode == 0x07) return 0x106;
    return 0;
}
static void input_poll_raw(void) {
    EFI_INPUT_KEY k;
    EFI_STATUS s;
    if (!ST->ConIn) return;
    for (;;) {
        s = ST->ConIn->ReadKeyStroke(ST->ConIn, &k);
        if (s & EFI_ERROR_MASK) break;
        UINT32 ev = input_map(k);
        if (ev == 0) continue;
        UINT32 next = (input_head + 1) % INPUT_QUEUE;
        if (next == input_tail) break;
        input_q[input_head] = ev;
        input_head = next;
    }
}
static UINT32 input_read(void) {
    input_poll_raw();
    if (input_tail == input_head) return 0;
    UINT32 ev = input_q[input_tail];
    input_tail = (input_tail + 1) % INPUT_QUEUE;
    return ev;
}

static char line[128];
static int line_len, line_cursor;
static UINT32 line_x, line_y;
static char history[8][128];
static int history_len[8];
static int history_n, history_pos;

static void line_redraw(void) {
    UINT32 i;
    con_x = line_x;
    con_y = line_y;
    for (i = 0; i < (UINT32)line_len; i++) con_putc(line[i]);
    for (i = line_len; i < 100; i++) con_putc(' ');
    con_x = line_x + (UINT32)line_cursor;
    con_y = line_y;
}
static void history_add(void) {
    int i;
    if (line_len == 0) return;
    for (i = 0; i < history_n; i++) {
        if (history_len[i] == line_len) {
            int j, same = 1;
            for (j = 0; j < line_len; j++) if (history[i][j] != line[j]) { same = 0; break; }
            if (same) return;
        }
    }
    if (history_n < 8) {
        for (i = 0; i < line_len; i++) history[history_n][i] = line[i];
        history_len[history_n] = line_len;
        history_n++;
    } else {
        for (i = 0; i < 7; i++) {
            int j;
            for (j = 0; j < 128; j++) history[i][j] = history[i + 1][j];
            history_len[i] = history_len[i + 1];
        }
        for (i = 0; i < line_len; i++) history[7][i] = line[i];
        history_len[7] = line_len;
    }
}
static void history_load(int idx) {
    int i;
    if (idx < 0 || idx >= history_n) return;
    line_len = history_len[idx];
    for (i = 0; i < line_len; i++) line[i] = history[idx][i];
    line_cursor = line_len;
    line_redraw();
}

static void run_command(void) {
    int i;
    line[line_len] = 0;
    con_puts("\r\n");
    if (line_len == 0) return;
    history_add();
    if (str_eq(line, "help")) show_help();
    else if (str_eq(line, "mem")) show_mem();
    else if (str_eq(line, "config")) show_config();
    else if (str_eq(line, "time")) show_time();
    else if (str_eq(line, "clear")) con_clear();
    else if (str_eq(line, "reboot") || str_eq(line, "poweroff")) {
        EFI_RUNTIME_SERVICES *rt = (EFI_RUNTIME_SERVICES *)ST->RuntimeServices;
        typedef void (EFIAPI *reset_fn)(UINT32, EFI_STATUS, UINTN, CHAR16 *);
        if (rt && rt->ResetSystem) {
            ((reset_fn)rt->ResetSystem)(str_eq(line, "reboot") ? 0 : 2, 0, 0, 0);
        }
        con_puts("reset unavailable\r\n");
    }
    else if (str_eq(line, "quit")) return;
    else if (str_prefix(line, "alloc ")) {
        UINT64 n = 0;
        int j = 6;
        while (line[j] >= '0' && line[j] <= '9') { n = n * 10 + (UINT64)(line[j] - '0'); j++; }
        void *p = heap_alloc((UINTN)n);
        con_puts("alloc ");
        con_dec(n);
        con_puts(" -> ");
        con_dec((UINT64)(UINTN)p);
        con_puts("\r\n");
    }
    else if (str_prefix(line, "echo ")) con_puts(line + 5), con_puts("\r\n");
    else {
        con_puts("unknown command: ");
        for (i = 0; i < line_len; i++) con_putc(line[i]);
        con_puts("\r\n");
    }
    con_puts("lepton> ");
    line_x = con_x; line_y = con_y;
}


/* Runtime compatibility stubs; UEFI uses lp_plat_* instead. */
void vga_cursor_left(void) { }
void vga_cursor_right(void) { }
void vga_clear(void) { con_clear(); }

/* ---- Lepton platform contract ------------------------------------------ */
static UINT8 *low_base;
static UINTN low_size, model_size;
static UINT32 platform_ticks;

static UINT32 time_ms(void) {
    EFI_RUNTIME_SERVICES *rt = (EFI_RUNTIME_SERVICES *)ST->RuntimeServices;
    EFI_TIME t;
    if (!rt || !rt->GetTime || (rt->GetTime(&t, 0) & EFI_ERROR_MASK)) return platform_ticks;
    return (((UINT32)t.Hour * 60 + t.Minute) * 60 + t.Second) * 1000
           + t.Nanosecond / 1000000;
}

#ifdef UEFI_KERNEL

static int wstr_eq(const CHAR16 *a, const CHAR16 *b) {
    while (*a && *b) { if (*a != *b) return 0; a++; b++; }
    return *a == *b;
}
static void wstr_copy(CHAR16 *d, const char *s) {
    while (*s) { *d++ = (CHAR16)(unsigned char)*s++; }
    *d = 0;
}
static EFI_STATUS open_model(EFI_FILE_PROTOCOL **out, UINT64 *size) {
    EFI_GUID loaded_guid = { 0x5B1B31A1, 0x9562, 0x11D2,
                             { 0x8E, 0x3F, 0x00, 0xA0, 0xC9, 0x69, 0x72, 0x3B } };
    EFI_GUID fs_guid = { 0x964E5B22, 0x6459, 0x11D2,
                         { 0x8E, 0x39, 0x00, 0xA0, 0xC9, 0x69, 0x72, 0x3B } };
    typedef EFI_STATUS (EFIAPI *handle_fn)(EFI_HANDLE, EFI_GUID *, void **);
    handle_fn hp = (handle_fn)ST->BootServices->HandleProtocol;
    EFI_LOADED_IMAGE_PROTOCOL *li = 0;
    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL *fs = 0;
    EFI_FILE_PROTOCOL *root = 0, *file = 0;
    EFI_STATUS s;
    CHAR16 path[128];
    s = hp(g_image, &loaded_guid, (void **)&li);
    if ((s & EFI_ERROR_MASK) || !li) return s;
    s = hp(li->DeviceHandle, &fs_guid, (void **)&fs);
    if ((s & EFI_ERROR_MASK) || !fs) return s;
    s = fs->OpenVolume(fs, &root);
    if ((s & EFI_ERROR_MASK) || !root) return s;
    wstr_copy(path, "\\models\\smollm2-instruct-q4_0.gguf");
    s = root->Open(root, &file, path, 1, 0);
    if (s & EFI_ERROR_MASK) {
        wstr_copy(path, "\\models\\tinystories.gguf");
        s = root->Open(root, &file, path, 1, 0);
    }
    if (s & EFI_ERROR_MASK) return s;
    s = file->SetPosition(file, 0xFFFFFFFFFFFFFFFFULL);
    if (s & EFI_ERROR_MASK) { file->Close(file); return s; }
    s = file->GetPosition(file, size);
    if (s & EFI_ERROR_MASK) { file->Close(file); return s; }
    s = file->SetPosition(file, 0);
    if (s & EFI_ERROR_MASK) { file->Close(file); return s; }
    *out = file;
    return 0;
}

static EFI_STATUS model_setup(void) {
    EFI_FILE_PROTOCOL *file = 0;
    UINT64 fsize = 0;
    UINTN want;
    void *pool = 0;
    EFI_STATUS s;
    s = open_model(&file, &fsize);
    if (s & EFI_ERROR_MASK) return s;
    if (fsize == 0 || fsize > 256ULL * 1024 * 1024) { file->Close(file); return 1; }
    model_size = (UINTN)fsize;
    want = ((UINTN)fsize + 66ULL * 1024 * 1024 + 0xFFFFF) & ~(UINTN)0xFFFFF;
    s = ST->BootServices->AllocatePool(4, want, &pool);
    if ((s & EFI_ERROR_MASK) || !pool) { file->Close(file); return 1; }
    low_base = (UINT8 *)pool;
    low_size = want;
    UINTN got = (UINTN)fsize;
    s = file->Read(file, &got, low_base);
    file->Close(file);
    if ((s & EFI_ERROR_MASK) || got != model_size) return 1;
    return (UINTN)low_base + low_size < 0xFFFFFFFFULL ? 0 : 1;
}

#endif

void lp_plat_serial_init(void) { serial_init(); }
void lp_plat_console_init(void) { con_clear(); }
void lp_plat_putc(char c) { con_putc(c); }
void lp_plat_interrupts_init(void) { }
void lp_plat_halt(void) { ST->BootServices->Stall(1000); }
UINT32 lp_plat_ticks(void) { platform_ticks = time_ms() / 10; return platform_ticks; }
UINT32 lp_plat_input_read(void) { return input_read(); }
void lp_plat_input_poll(void) { input_poll_raw(); }
UINT32 lp_plat_input_cancel_pending(void) {
    input_poll_raw();
    UINT32 i = input_tail;
    while (i != input_head) {
        if (input_q[i] == 3) return 1;
        i = (i + 1) % INPUT_QUEUE;
    }
    return 0;
}
void lp_plat_cursor_left(void) { if (con_x > 0) con_x--; }
void lp_plat_cursor_right(void) { con_x++; }
void lp_plat_clear_screen(void) { con_clear(); }
void lp_plat_set_color(uint8_t c) {
    if (c == 0x0A) fg_color = 0x0000FF00;
    else if (c == 0x0C) fg_color = 0x00FF0000;
    else fg_color = 0x00FFFFFF;
}
void lp_plat_page_up(void) { }
void lp_plat_page_down(void) { }
UINT32 lp_plat_module_count(void) {
#ifdef UEFI_KERNEL
    return low_base ? 1 : 0;
#else
    return 0;
#endif
}
UINT32 lp_plat_module_start(uint32_t idx) {
#ifdef UEFI_KERNEL
    (void)idx;
    return (UINT32)(UINTN)low_base;
#else
    (void)idx; return 0;
#endif
}
UINT32 lp_plat_module_size(uint32_t idx) {
#ifdef UEFI_KERNEL
    (void)idx;
    return (UINT32)model_size;
#else
    (void)idx; return 0;
#endif
}
UINT32 lp_plat_module_end(uint32_t idx) {
#ifdef UEFI_KERNEL
    (void)idx;
    return (UINT32)((UINTN)low_base + model_size);
#else
    (void)idx; return 0;
#endif
}
UINT32 lp_plat_memory_end(void) {
#ifdef UEFI_KERNEL
    return (UINT32)((UINTN)low_base + low_size);
#else
    return 0;
#endif
}
void lp_plat_reboot(void) {
    EFI_RUNTIME_SERVICES *rt = (EFI_RUNTIME_SERVICES *)ST->RuntimeServices;
    typedef void (EFIAPI *reset_fn)(UINT32, EFI_STATUS, UINTN, CHAR16 *);
    if (rt && rt->ResetSystem) ((reset_fn)rt->ResetSystem)(0, 0, 0, 0);
}
void lp_plat_poweroff(void) {
    EFI_RUNTIME_SERVICES *rt = (EFI_RUNTIME_SERVICES *)ST->RuntimeServices;
    typedef void (EFIAPI *reset_fn)(UINT32, EFI_STATUS, UINTN, CHAR16 *);
    if (rt && rt->ResetSystem) ((reset_fn)rt->ResetSystem)(2, 0, 0, 0);
}

EFI_STATUS EFIAPI efi_main(EFI_HANDLE image, EFI_SYSTEM_TABLE *st) {
    EFI_STATUS s;
    EFI_GUID gop_guid = { 0x9042A9DE, 0x23DC, 0x4A38,
                          { 0x96, 0xFB, 0x7A, 0xDE, 0xD0, 0x80, 0x51, 0x6A } };
    static UINT8 map_buf[16384];
    UINTN map_size = sizeof(map_buf), map_key = 0;
    UINT32 map_ver = 0;
    g_image = image;
    ST = st;
    serial_init();
    serial_puts("Lepton UEFI64 shell\n");
    if (!ST || !ST->BootServices || !ST->ConIn) return 1;

    s = ST->BootServices->LocateProtocol(&gop_guid, 0, (void **)&gop);
    if ((s & EFI_ERROR_MASK) || !gop || !gop->Mode || !gop->Mode->Info) {
        serial_puts("GOP unavailable\n");
        return 1;
    }
    fb = (UINT8 *)(UINTN)gop->Mode->FrameBufferBase;
    fb_width = gop->Mode->Info->HorizontalResolution;
    fb_height = gop->Mode->Info->VerticalResolution;
    fb_pitch = gop->Mode->Info->PixelsPerScanLine * 4;
    fb_format = gop->Mode->Info->PixelFormat;
    con_cols = fb_width / 8;
    con_rows = fb_height / 16;
    con_clear();

    s = ST->BootServices->GetMemoryMap(&map_size, (EFI_MEMORY_DESCRIPTOR *)map_buf,
                                       &map_key, &map_desc_size, &map_ver);
    if ((s & EFI_ERROR_MASK) == 0 && map_desc_size != 0) {
        UINTN i;
        UINT8 *p = map_buf;
        map_count = map_size / map_desc_size;
        for (i = 0; i < map_count; i++) {
            EFI_MEMORY_DESCRIPTOR *d = (EFI_MEMORY_DESCRIPTOR *)p;
            map_pages += d->NumberOfPages;
            p += map_desc_size;
        }
    }

#ifdef UEFI_KERNEL
    mirror_serial = 0;
    if (model_setup() != 0) {
        con_puts("model setup failed\r\n");
        return 1;
    }
    con_clear();
    main();
    return EFI_SUCCESS;
#else
    con_puts("Lepton UEFI64 platform shell\r\n");
    con_puts("--------------------------------\r\n");
    show_config();
    if (heap_init(16 * 1024 * 1024) & EFI_ERROR_MASK) {
        con_puts("heap init failed\r\n");
        return 1;
    }
    show_mem();
    con_puts("type help\r\n\r\n");
    con_puts("lepton> ");
    line_x = con_x; line_y = con_y;

    for (;;) {
        UINT32 k = input_read();
        if (k == 0) { ST->BootServices->Stall(10000); continue; }
        if (k == 13 || k == 10) {
            run_command();
            if (line_len == 4 && line[0] == 'q' && line[1] == 'u' && line[2] == 'i' && line[3] == 't') {
                return 0;
            }
            line_len = 0; line_cursor = 0; history_pos = history_n;
        } else if (k == 8) {
            if (line_cursor > 0) {
                int i;
                for (i = line_cursor - 1; i < line_len - 1; i++) line[i] = line[i + 1];
                line_len--; line_cursor--; line_redraw();
            }
        } else if (k == 9) {
            if (line_cursor < line_len) {
                int i;
                for (i = line_len; i > line_cursor; i--) line[i] = line[i - 1];
                line[line_cursor] = ' ';
                line_len++; line_cursor++; line_redraw();
            }
        } else if (k == 0x100) {
            if (history_pos > 0) { history_pos--; history_load(history_pos); }
        } else if (k == 0x101) {
            if (history_pos < history_n) { history_pos++; if (history_pos == history_n) { line_len = 0; line_cursor = 0; line_redraw(); } else history_load(history_pos); }
        } else if (k == 0x102) {
            if (line_cursor > 0) { line_cursor--; line_redraw(); }
        } else if (k == 0x103) {
            if (line_cursor < line_len) { line_cursor++; line_redraw(); }
        } else if (k == 0x104) {
            line_cursor = 0; line_redraw();
        } else if (k == 0x105) {
            line_cursor = line_len; line_redraw();
        } else if (k == 0x106) {
            if (line_cursor < line_len) {
                int i;
                for (i = line_cursor; i < line_len - 1; i++) line[i] = line[i + 1];
                line_len--; line_redraw();
            }
        } else if (k >= 32 && k < 127 && line_len < 100) {
            int i;
            for (i = line_len; i > line_cursor; i--) line[i] = line[i - 1];
            line[line_cursor] = (char)k;
            line_len++; line_cursor++; line_redraw();
        }
    }
#endif
}
