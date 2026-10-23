/* Minimal UEFI64 application: serial banner, GOP info, memory map. */
#include "uefi.h"

static EFI_SYSTEM_TABLE *ST;


static inline void outb(UINT16 port, UINT8 v) {
    __asm__ volatile("outb %0, %1" : : "a"(v), "Nd"(port));
}
static inline UINT8 inb(UINT16 port) {
    UINT8 v;
    __asm__ volatile("inb %1, %0" : "=a"(v) : "Nd"(port));
    return v;
}
static void serial_init(void) {
    outb(0x3F8 + 1, 0x00);
    outb(0x3F8 + 3, 0x80);
    outb(0x3F8 + 0, 0x03);
    outb(0x3F8 + 1, 0x00);
    outb(0x3F8 + 3, 0x03);
    outb(0x3F8 + 2, 0xC7);
    outb(0x3F8 + 4, 0x0B);
}
static void serial_putc(char c) {
    while ((inb(0x3F8 + 5) & 0x20) == 0) { }
    outb(0x3F8, (UINT8)c);
}
static void serial_puts(const char *s) {
    while (*s) { serial_putc(*s++); }
}
static void serial_hex(UINT64 v) {
    const char *h = "0123456789ABCDEF";
    serial_puts("0x");
    for (int i = 60; i >= 0; i -= 4) { serial_putc(h[(v >> i) & 0xF]); }
}
static void serial_dec(UINT64 v) {
    char b[24];
    int n = 0;
    if (v == 0) { serial_putc('0'); return; }
    while (v > 0 && n < 23) { b[n++] = (char)('0' + (v % 10)); v /= 10; }
    while (n > 0) { serial_putc(b[--n]); }
}

static void uefi_print(const char *s) {
    CHAR16 buf[160];
    int i = 0;
    while (s[i] && i < 158) { buf[i] = (CHAR16)(unsigned char)s[i]; i++; }
    buf[i] = 0;
    if (ST && ST->ConOut) { ST->ConOut->OutputString(ST->ConOut, buf); }
}

static void uefi_dec(UINT64 v) {
    CHAR16 b[24];
    int n = 0;
    if (v == 0) { b[n++] = '0'; }
    while (v > 0 && n < 23) { b[n++] = (CHAR16)('0' + (v % 10)); v /= 10; }
    CHAR16 out[24];
    int k = 0;
    while (n > 0) { out[k++] = b[--n]; }
    out[k] = 0;
    if (ST && ST->ConOut) { ST->ConOut->OutputString(ST->ConOut, out); }
}

static const EFI_GUID gop_guid = {
    0x9042A9DE, 0x23DC, 0x4A38,
    { 0x96, 0xFB, 0x7A, 0xDE, 0xD0, 0x80, 0x51, 0x6A }
};

EFI_STATUS EFIAPI efi_main(EFI_HANDLE image_handle, EFI_SYSTEM_TABLE *system_table) {
    (void)image_handle;
    ST = system_table;
    serial_init();
    serial_puts("Lepton UEFI64 hello\n");
    uefi_print("Lepton UEFI64 hello\r\n");

    if (!ST || !ST->BootServices) {
        serial_puts("no boot services\n");
        return 1;
    }

    EFI_BOOT_SERVICES *bs = ST->BootServices;
    static UINT8 map_buf[16384];
    UINTN map_size = sizeof(map_buf);
    UINTN map_key = 0, desc_size = 0;
    UINT32 desc_ver = 0;
    EFI_STATUS st = bs->GetMemoryMap(&map_size, (EFI_MEMORY_DESCRIPTOR *)map_buf,
                                     &map_key, &desc_size, &desc_ver);
    serial_puts("GetMemoryMap status ");
    serial_hex(st);
    serial_puts("\n");
    if ((st & EFI_ERROR_MASK) == 0 && desc_size != 0) {
        UINTN count = map_size / desc_size;
        UINT64 pages = 0;
        UINT8 *p = map_buf;
        for (UINTN i = 0; i < count; i++) {
            EFI_MEMORY_DESCRIPTOR *d = (EFI_MEMORY_DESCRIPTOR *)p;
            pages += d->NumberOfPages;
            p += desc_size;
        }
        serial_puts("descriptors ");
        serial_dec(count);
        serial_puts(", pages ");
        serial_dec(pages);
        serial_puts(", bytes ");
        serial_dec(pages * 4096ULL);
        serial_puts("\n");
        uefi_print("Memory map: descriptors ");
        uefi_dec(count);
        uefi_print(", pages ");
        uefi_dec(pages);
        uefi_print("\r\n");
    } else {
        serial_puts("memory map query failed/needs a larger buffer\n");
    }

    EFI_GRAPHICS_OUTPUT_PROTOCOL *gop = 0;
    EFI_STATUS gs = bs->LocateProtocol((EFI_GUID *)&gop_guid, 0, (void **)&gop);
    serial_puts("LocateProtocol(GOP) status ");
    serial_hex(gs);
    serial_puts("\n");
    if ((gs & EFI_ERROR_MASK) == 0 && gop && gop->Mode && gop->Mode->Info) {
        serial_puts("GOP ");
        serial_dec(gop->Mode->Info->HorizontalResolution);
        serial_puts("x");
        serial_dec(gop->Mode->Info->VerticalResolution);
        serial_puts(", fb ");
        serial_hex(gop->Mode->FrameBufferBase);
        serial_puts(", size ");
        serial_dec(gop->Mode->FrameBufferSize);
        serial_puts("\n");
    } else {
        serial_puts("GOP not available\n");
    }
    if ((gs & EFI_ERROR_MASK) == 0 && gop && gop->Mode && gop->Mode->Info) {
        uefi_print("GOP ");
        uefi_dec(gop->Mode->Info->HorizontalResolution);
        uefi_print("x");
        uefi_dec(gop->Mode->Info->VerticalResolution);
        uefi_print("\r\n");
    }

    serial_puts("Lepton UEFI64 done\n");
    uefi_print("Lepton UEFI64 done\r\n");
    return EFI_SUCCESS;
}
