/* BIOS32 implementation of the shared platform contract. */
#include "../../platform.h"

extern void serial_init(void);
extern void vga_init(void);
extern void vga_putc(char c);
extern void interrupts_init(void);
extern void cpu_hlt(void);
extern uint32_t ticks(void);
extern uint32_t kbd_read(void);
extern void kbd_poll(void);
extern uint32_t kbd_cancel_pending(void);
extern void lp_cursor_left(void);
extern void lp_cursor_right(void);
extern void lp_clear_screen(void);
extern void vga_set_color(uint8_t c);
extern void vga_page_up(void);
extern void vga_page_down(void);
extern uint32_t mb_module_count(void);
extern uint32_t mb_module_start(uint32_t idx);
extern uint32_t mb_module_size(uint32_t idx);
extern uint32_t mb_module_end(uint32_t idx);
extern uint32_t mb_memory_end(void);
extern void lp_reboot(void);
extern void lp_poweroff(void);

void lp_plat_serial_init(void) { serial_init(); }
void lp_plat_console_init(void) { vga_init(); }
void lp_plat_putc(char c) { vga_putc(c); }
void lp_plat_interrupts_init(void) { interrupts_init(); }
void lp_plat_halt(void) { cpu_hlt(); }
uint32_t lp_plat_ticks(void) { return ticks(); }
uint32_t lp_plat_input_read(void) { return kbd_read(); }
void lp_plat_input_poll(void) { kbd_poll(); }
uint32_t lp_plat_input_cancel_pending(void) { return kbd_cancel_pending(); }
void lp_plat_cursor_left(void) { lp_cursor_left(); }
void lp_plat_cursor_right(void) { lp_cursor_right(); }
void lp_plat_clear_screen(void) { lp_clear_screen(); }
void lp_plat_set_color(uint8_t c) { vga_set_color(c); }
void lp_plat_page_up(void) { vga_page_up(); }
void lp_plat_page_down(void) { vga_page_down(); }
uint32_t lp_plat_module_count(void) { return mb_module_count(); }
uint32_t lp_plat_module_start(uint32_t idx) { return mb_module_start(idx); }
uint32_t lp_plat_module_size(uint32_t idx) { return mb_module_size(idx); }
uint32_t lp_plat_module_end(uint32_t idx) { return mb_module_end(idx); }
uint32_t lp_plat_memory_end(void) { return mb_memory_end(); }
void lp_plat_reboot(void) { lp_reboot(); }
void lp_plat_poweroff(void) { lp_poweroff(); }
