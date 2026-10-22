/* VGA text-mode driver (80x25, 16 colors).
   Writes directly to physical memory 0xB8000. */

#define VGA_WIDTH  80
#define VGA_HEIGHT 25
#define VGA_MEMORY 0xB8000

static volatile unsigned short* vga_buffer = (unsigned short*)VGA_MEMORY;
static int cursor_x = 0;
static int cursor_y = 0;
static unsigned char color = 0x0F;  /* White on black */

#define SCROLL_PAGES 8
static unsigned short scroll_history[SCROLL_PAGES][VGA_WIDTH * VGA_HEIGHT];
static unsigned short live_page[VGA_WIDTH * VGA_HEIGHT];
static int scroll_head = 0;
static int scroll_count = 0;
static int scroll_pos = 0;
static int scroll_active = 0;

static inline void outb_vga(unsigned short port, unsigned char val) {
    __asm__ volatile("outb %0, %1" : : "a"(val), "Nd"(port));
}

static void vga_update_cursor(void) {
    unsigned short pos = cursor_y * VGA_WIDTH + cursor_x;
    outb_vga(0x3D4, 0x0F);
    outb_vga(0x3D5, (unsigned char)(pos & 0xFF));
    outb_vga(0x3D4, 0x0E);
    outb_vga(0x3D5, (unsigned char)((pos >> 8) & 0xFF));
}

static void vga_copy_page(volatile unsigned short *dst, volatile unsigned short *src) {
    for (int i = 0; i < VGA_WIDTH * VGA_HEIGHT; i++) { dst[i] = src[i]; }
}

static void vga_show_scroll_page(void) {
    if (scroll_pos == 0) {
        vga_copy_page(vga_buffer, live_page);
        scroll_active = 0;
    } else {
        int idx = (scroll_head - scroll_pos + SCROLL_PAGES) % SCROLL_PAGES;
        vga_copy_page(vga_buffer, scroll_history[idx]);
        scroll_active = 1;
    }
    vga_update_cursor();
}

void vga_page_up(void) {
    if (!scroll_active) {
        vga_copy_page(live_page, vga_buffer);
        scroll_active = 1;
        scroll_pos = 0;
    }
    if (scroll_pos < scroll_count) {
        scroll_pos = scroll_pos + 1;
        vga_show_scroll_page();
    }
}

void vga_page_down(void) {
    if (!scroll_active) { return; }
    if (scroll_pos > 0) {
        scroll_pos = scroll_pos - 1;
        vga_show_scroll_page();
    }
}

void vga_scroll_exit(void) {
    if (scroll_active) {
        vga_copy_page(vga_buffer, live_page);
        scroll_active = 0;
        scroll_pos = 0;
        vga_update_cursor();
    }
}

static void vga_scroll(void) {
    /* Keep the page being pushed off-screen in the scrollback ring. */
    vga_copy_page(scroll_history[scroll_head], vga_buffer);
    scroll_head = (scroll_head + 1) % SCROLL_PAGES;
    if (scroll_count < SCROLL_PAGES) { scroll_count = scroll_count + 1; }
    scroll_pos = 0;
    scroll_active = 0;
    for (int i = 0; i < VGA_HEIGHT - 1; i++) {
        for (int j = 0; j < VGA_WIDTH; j++) {
            vga_buffer[i * VGA_WIDTH + j] = vga_buffer[(i + 1) * VGA_WIDTH + j];
        }
    }
    for (int j = 0; j < VGA_WIDTH; j++) {
        vga_buffer[(VGA_HEIGHT - 1) * VGA_WIDTH + j] = (color << 8) | ' ';
    }
}

void vga_clear(void) {
    scroll_count = 0;
    scroll_head = 0;
    scroll_pos = 0;
    scroll_active = 0;
    for (int i = 0; i < VGA_WIDTH * VGA_HEIGHT; i++) {
        vga_buffer[i] = (color << 8) | ' ';
    }
    cursor_x = 0;
    cursor_y = 0;
    vga_update_cursor();
}

void vga_putc(char c) {
    vga_scroll_exit();
    if (c == '\n') {
        cursor_x = 0;
        cursor_y++;
    } else if (c == '\r') {
        cursor_x = 0;
    } else if (c == '\b') {
        if (cursor_x > 0) {
            cursor_x--;
        } else if (cursor_y > 0) {
            cursor_y--;
            cursor_x = VGA_WIDTH - 1;
        }
        vga_buffer[cursor_y * VGA_WIDTH + cursor_x] = (color << 8) | ' ';
    } else {
        vga_buffer[cursor_y * VGA_WIDTH + cursor_x] = (color << 8) | c;
        cursor_x++;
        if (cursor_x >= VGA_WIDTH) {
            cursor_x = 0;
            cursor_y++;
        }
    }

    if (cursor_y >= VGA_HEIGHT) {
        vga_scroll();
        cursor_y = VGA_HEIGHT - 1;
    }
    vga_update_cursor();
}

void vga_set_color(unsigned char c) {
    color = c;
}

void vga_cursor_left(void) {
    vga_scroll_exit();
    if (cursor_x > 0) {
        cursor_x--;
    } else if (cursor_y > 0) {
        cursor_y--;
        cursor_x = VGA_WIDTH - 1;
    }
    vga_update_cursor();
}

void vga_cursor_right(void) {
    vga_scroll_exit();
    cursor_x++;
    if (cursor_x >= VGA_WIDTH) {
        cursor_x = 0;
        cursor_y++;
    }
    if (cursor_y >= VGA_HEIGHT) {
        vga_scroll();
        cursor_y = VGA_HEIGHT - 1;
    }
    vga_update_cursor();
}

void vga_init(void) {
    vga_clear();
}
