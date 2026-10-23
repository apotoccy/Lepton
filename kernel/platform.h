/* Shared platform contract for BIOS32 and UEFI64 builds. */
#ifndef LEPTON_PLATFORM_H
#define LEPTON_PLATFORM_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    uintptr_t memory_start;
    uintptr_t memory_end;
    unsigned char *framebuffer;
    uint32_t framebuffer_width;
    uint32_t framebuffer_height;
    uint32_t framebuffer_pitch;
    uintptr_t module_start;
    uint32_t module_size;
    void *acpi_rsdp;
    void *boot_services;
} lp_boot_info;

void lp_plat_serial_init(void);
void lp_plat_console_init(void);
void lp_plat_putc(char c);
void lp_plat_interrupts_init(void);
void lp_plat_halt(void);
uint32_t lp_plat_ticks(void);
uint32_t lp_plat_input_read(void);
void lp_plat_input_poll(void);
uint32_t lp_plat_input_cancel_pending(void);
void lp_plat_cursor_left(void);
void lp_plat_cursor_right(void);
void lp_plat_clear_screen(void);
void lp_plat_set_color(uint8_t c);
void lp_plat_page_up(void);
void lp_plat_page_down(void);
uint32_t lp_plat_module_count(void);
uint32_t lp_plat_module_start(uint32_t idx);
uint32_t lp_plat_module_size(uint32_t idx);
uint32_t lp_plat_module_end(uint32_t idx);
uint32_t lp_plat_memory_end(void);
void lp_plat_reboot(void);
void lp_plat_poweroff(void);

#endif
