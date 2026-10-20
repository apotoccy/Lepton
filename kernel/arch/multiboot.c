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
    uint32_t syms[4];
    uint32_t mmap_length;
    uint32_t mmap_addr;
};

struct mb_mmap_entry {
    uint32_t size;
    uint64_t addr;
    uint64_t len;
    uint32_t type;
} __attribute__((packed));

struct mb_module_t {
    uint32_t mod_start;
    uint32_t mod_end;
    uint32_t string;
    uint32_t reserved;
};

#define MB_FLAG_MEM  0x01u
#define MB_FLAG_MODS 0x08u
#define MB_FLAG_MMAP 0x40u

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

/* Highest usable physical address from the Multiboot memory map. */
uint32_t mb_memory_end(void) {
    struct mb_info_t *info = (struct mb_info_t *)mb_info;
    uint32_t best = 0;

    if ((info->flags & MB_FLAG_MMAP) != 0
            && info->mmap_length != 0 && info->mmap_addr != 0) {
        unsigned char *p = (unsigned char *)info->mmap_addr;
        unsigned char *end = p + info->mmap_length;
        while (p + 4 <= end) {
            struct mb_mmap_entry *e = (struct mb_mmap_entry *)p;
            if (e->size < 20) break;
            if (e->type == 1) {
                uint64_t last = e->addr + e->len;
                uint32_t last32 = last > 0xFFFFFFFFULL
                                ? 0xFFFFFFFFu : (uint32_t)last;
                if (last32 > best) best = last32;
            }
            p += e->size + 4;
        }
        if (best != 0) return best;
    }

    if ((info->flags & MB_FLAG_MEM) != 0) {
        uint32_t end = 0x100000u + info->mem_upper * 1024u;
        if (end > best) best = end;
    }
    return best;
}
