/* Multiboot 1 module access.
 *
 * QEMU's `-initrd` flag and GRUB's `module` directive both populate the
 * multiboot info's module list. Each entry is (physical start, physical
 * end) of a file loaded into memory. Since the kernel runs in flat
 * 32-bit protected mode with no paging, these physical addresses are
 * directly readable as pointers.
 *
 * Module data lives in memory owned by the bootloader, not by our heap,
 * so reading it costs nothing and there is nothing to free. */

#include <stdint.h>

extern uint32_t mb_magic;   /* stashed by boot.S from %eax */
extern uint32_t mb_info;    /* stashed by boot.S from %ebx */

/* Multiboot 1 info block. We only need up to mods_addr; the full
   structure has additional fields past this point. */
struct mb_info_t {
    uint32_t flags;
    uint32_t mem_lower;
    uint32_t mem_upper;
    uint32_t boot_device;
    uint32_t cmdline;
    uint32_t mods_count;
    uint32_t mods_addr;
};

struct mb_module_t {
    uint32_t mod_start;
    uint32_t mod_end;
    uint32_t string;
    uint32_t reserved;
};

#define MB_FLAG_MODS 0x08u

uint32_t mb_module_count(void) {
    struct mb_info_t *info = (struct mb_info_t *)mb_info;
    if ((info->flags & MB_FLAG_MODS) == 0) return 0;
    return info->mods_count;
}

uint32_t mb_module_start(uint32_t idx) {
    struct mb_info_t *info = (struct mb_info_t *)mb_info;
    struct mb_module_t *mods = (struct mb_module_t *)info->mods_addr;
    return mods[idx].mod_start;
}

uint32_t mb_module_size(uint32_t idx) {
    struct mb_info_t *info = (struct mb_info_t *)mb_info;
    struct mb_module_t *mods = (struct mb_module_t *)info->mods_addr;
    return mods[idx].mod_end - mods[idx].mod_start;
}

uint32_t mb_module_end(uint32_t idx) {
    struct mb_info_t *info = (struct mb_info_t *)mb_info;
    struct mb_module_t *mods = (struct mb_module_t *)info->mods_addr;
    return mods[idx].mod_end;
}
