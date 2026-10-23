/* Minimal UEFI64 application: serial banner, GOP info, memory map.
   No gnu-efi dependency; only the UEFI ABI subset needed here. */
#include <stdint.h>
#include <stddef.h>

typedef uint8_t  UINT8;
typedef uint16_t UINT16;
typedef uint32_t UINT32;
typedef uint64_t UINT64;
typedef uint64_t UINTN;
typedef uint16_t CHAR16;
typedef uint64_t EFI_STATUS;
typedef void    *EFI_HANDLE;
typedef uint64_t EFI_PHYSICAL_ADDRESS;
typedef uint64_t EFI_VIRTUAL_ADDRESS;

#define EFIAPI __attribute__((ms_abi))
#define EFI_SUCCESS 0
#define EFI_ERROR_MASK 0x8000000000000000ULL
#define EFI_BUFFER_TOO_SMALL 0x8000000000000005ULL

typedef struct {
    UINT32 Data1;
    UINT16 Data2;
    UINT16 Data3;
    UINT8  Data4[8];
} EFI_GUID;

typedef struct {
    UINT64 Signature;
    UINT32 Revision;
    UINT32 HeaderSize;
    UINT32 CRC32;
    UINT32 Reserved;
} EFI_TABLE_HEADER;

typedef struct {
    UINT32 Type;
    UINT32 Pad;
    EFI_PHYSICAL_ADDRESS PhysicalStart;
    EFI_VIRTUAL_ADDRESS VirtualStart;
    UINT64 NumberOfPages;
    UINT64 Attribute;
} EFI_MEMORY_DESCRIPTOR;

typedef struct {
    UINT32 Version;
    UINT32 HorizontalResolution;
    UINT32 VerticalResolution;
    UINT32 PixelFormat;
    UINT32 PixelInformation[4];
    UINT32 PixelsPerScanLine;
} EFI_GRAPHICS_OUTPUT_MODE_INFORMATION;

typedef struct {
    UINT32 MaxMode;
    UINT32 Mode;
    EFI_GRAPHICS_OUTPUT_MODE_INFORMATION *Info;
    UINTN SizeOfInfo;
    EFI_PHYSICAL_ADDRESS FrameBufferBase;
    UINTN FrameBufferSize;
} EFI_GRAPHICS_OUTPUT_PROTOCOL_MODE;

typedef struct {
    EFI_STATUS (EFIAPI *QueryMode)(void *This, UINT32 ModeNumber,
                                   UINTN *SizeOfInfo,
                                   EFI_GRAPHICS_OUTPUT_MODE_INFORMATION **Info);
    EFI_STATUS (EFIAPI *SetMode)(void *This, UINT32 ModeNumber);
    EFI_STATUS (EFIAPI *Blt)(void *This, void *BltBuffer,
                             UINT32 BltOperation, UINTN SourceX,
                             UINTN SourceY, UINTN DestinationX,
                             UINTN DestinationY, UINTN Width,
                             UINTN Height, UINTN Delta);
    EFI_GRAPHICS_OUTPUT_PROTOCOL_MODE *Mode;
} EFI_GRAPHICS_OUTPUT_PROTOCOL;

typedef struct {
    EFI_STATUS (EFIAPI *Reset)(void *This, UINT8 ExtendedVerification);
    EFI_STATUS (EFIAPI *OutputString)(void *This, CHAR16 *String);
    EFI_STATUS (EFIAPI *TestString)(void *This, CHAR16 *String);
    EFI_STATUS (EFIAPI *QueryMode)(void *This, UINTN ModeNumber,
                                   UINTN *Columns, UINTN *Rows);
    EFI_STATUS (EFIAPI *SetMode)(void *This, UINTN ModeNumber);
    EFI_STATUS (EFIAPI *SetAttribute)(void *This, UINTN Attribute);
    EFI_STATUS (EFIAPI *ClearScreen)(void *This);
    EFI_STATUS (EFIAPI *SetCursorPosition)(void *This, UINTN Column, UINTN Row);
    EFI_STATUS (EFIAPI *EnableCursor)(void *This, UINT8 Visible);
    void *Mode;
} EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL;

typedef struct {
    UINT64 Signature;
    UINT32 Revision;
    UINT32 HeaderSize;
    UINT32 CRC32;
    UINT32 Reserved;
    CHAR16 *FirmwareVendor;
    UINT32 FirmwareRevision;
    EFI_HANDLE ConsoleInHandle;
    void *ConIn;
    EFI_HANDLE ConsoleOutHandle;
    EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *ConOut;
    EFI_HANDLE StandardErrorHandle;
    EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *StdErr;
    void *RuntimeServices;
    struct EFI_BOOT_SERVICES *BootServices;
    UINTN NumberOfTableEntries;
    void *ConfigurationTable;
} EFI_SYSTEM_TABLE;

typedef struct EFI_BOOT_SERVICES {
    EFI_TABLE_HEADER Hdr;
    void *RaiseTPL;
    void *RestoreTPL;
    EFI_STATUS (EFIAPI *AllocatePages)(UINT32 Type, UINT32 MemoryType,
                                       UINTN Pages, EFI_PHYSICAL_ADDRESS *Memory);
    EFI_STATUS (EFIAPI *FreePages)(EFI_PHYSICAL_ADDRESS Memory, UINTN Pages);
    EFI_STATUS (EFIAPI *GetMemoryMap)(UINTN *MemoryMapSize,
                                      EFI_MEMORY_DESCRIPTOR *MemoryMap,
                                      UINTN *MapKey, UINTN *DescriptorSize,
                                      UINT32 *DescriptorVersion);
    void *AllocatePool;
    void *FreePool;
    void *CreateEvent;
    void *SetTimer;
    void *WaitForEvent;
    void *SignalEvent;
    void *CloseEvent;
    void *CheckEvent;
    void *InstallProtocolInterface;
    void *ReinstallProtocolInterface;
    void *UninstallProtocolInterface;
    void *HandleProtocol;
    void *Reserved;
    void *RegisterProtocolNotify;
    void *LocateHandle;
    void *LocateDevicePath;
    void *InstallConfigurationTable;
    void *LoadImage;
    void *StartImage;
    void *Exit;
    void *UnloadImage;
    void *ExitBootServices;
    void *GetNextMonotonicCount;
    void *Stall;
    void *SetWatchdogTimer;
    void *ConnectController;
    void *DisconnectController;
    void *OpenProtocol;
    void *CloseProtocol;
    void *OpenProtocolInformation;
    void *ProtocolsPerHandle;
    void *LocateHandleBuffer;
    EFI_STATUS (EFIAPI *LocateProtocol)(EFI_GUID *Protocol,
                                        void *Registration,
                                        void **Interface);
    void *InstallMultipleProtocolInterfaces;
    void *UninstallMultipleProtocolInterfaces;
    void *CalculateCrc32;
    void *CopyMem;
    void *SetMem;
    void *CreateEventEx;
} EFI_BOOT_SERVICES;

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
