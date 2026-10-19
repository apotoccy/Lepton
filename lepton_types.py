"""
Lepton type system.

Everything related to primitive types, their C lowering, and the small
helpers the checker uses. Kept deliberately separate from leptonc.py so
that Phase 1's numeric work (f16, bf16, q15, SIMD vectors, per-target
capability tables) has a natural home.
"""

import math
import struct

PRIMS = {'i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64',
         'f16', 'bf16', 'f32', 'f64',
         'q7', 'q15', 'q31',
         'v2f64', 'v4f32', 'v4i32', 'v8i16', 'v16i8',
         'bool', 'char', 'str', 'void'}

CTYPE = {
    'i8':  'int8_t',   'i16': 'int16_t', 'i32': 'int32_t', 'i64': 'int64_t',
    'u8':  'uint8_t',  'u16': 'uint16_t','u32': 'uint32_t','u64': 'uint64_t',
    'f16': 'lp_f16',   'bf16': 'lp_bf16',
    'f32': 'float',    'f64': 'double',
    'v2f64': 'lp_v2f64', 'v4f32': 'lp_v4f32', 'v4i32': 'lp_v4i32',
    'v8i16': 'lp_v8i16', 'v16i8': 'lp_v16i8',
    'q7':  'lp_q7',    'q15': 'lp_q15',  'q31': 'lp_q31',
    'bool': 'bool',    'char': 'char',   'void': 'void',   'str': 'const char*',
}

RANK = {
    'bool': 0,
    'char': 1, 'i8': 1,
    'u8':   2,
    'i16':  3, 'u16': 4,
    'i32':  5, 'u32': 6,
    'i64':  7, 'u64': 8,
    'q7':   9, 'q15': 10, 'q31': 11,
    'bf16': 12,
    'f16':  13,
    'f32':  14, 'f64': 15,
}


def is_int(t):
    return t[0] == 'prim' and t[1] in ('i8', 'i16', 'i32', 'i64',
                                       'u8', 'u16', 'u32', 'u64', 'char')


def is_quant(t):
    return t[0] == 'prim' and t[1] in ('q7', 'q15', 'q31')
    
VECTOR_INFO = {
    'v2f64': ('f64',  2),
    'v4f32': ('f32',  4),
    'v4i32': ('i32',  4),
    'v8i16': ('i16',  8),
    'v16i8': ('i8',  16),
}


def is_vector(t):
    return t[0] == 'prim' and t[1] in VECTOR_INFO


def vector_elem(t):
    return ('prim', VECTOR_INFO[t[1]][0])


def vector_len(t):
    return VECTOR_INFO[t[1]][1]


def vector_of(elem_name, n):
    """Return the vector type for n elements of a primitive type, or None."""
    for vt, (et, w) in VECTOR_INFO.items():
        if et == elem_name and w == n:
            return ('prim', vt)
    return None
    
def is_array(t):
    return t[0] == 'array'


def array_elem(t):
    return t[1]


def array_len(t):
    return t[2]


def array_ctype_name(elem_type, count):
    """Stable C name for a structurally-typed array: 'lp_arr_f32_16'.

    Pointer element types include their pointee so that arrays such as
    [ptr[u8]; 4] and [ptr[f32]; 4] do not collide.
    """
    def inner_name(t):
        if t[0] == 'prim':
            return t[1]
        if t[0] == 'named':
            return t[1]
        if t[0] == 'array':
            return array_ctype_name(t[1], t[2])
        if t[0] == 'ptr':
            return 'ptr_' + inner_name(t[1])
        return 'unknown'

    return f"lp_arr_{inner_name(elem_type)}_{count}"

def is_num(t):
    return is_int(t) or is_quant(t) or \
           (t[0] == 'prim' and t[1] in ('f16', 'bf16', 'f32', 'f64'))


# Width and signedness for integer primitives. `char` is treated as an
# 8-bit signed value for conversion diagnostics.
INT_INFO = {
    'i8':  (8, True),
    'i16': (16, True),
    'i32': (32, True),
    'i64': (64, True),
    'u8':  (8, False),
    'u16': (16, False),
    'u32': (32, False),
    'u64': (64, False),
    'char': (8, True),
}

FLOAT_NAMES = {'f16', 'bf16', 'f32', 'f64'}
QUANT_NAMES = {'q7', 'q15', 'q31'}


def is_lossy_conversion(src, dst):
    """True when an implicit numeric conversion can lose information.

    Policy:
      - integer widening (and unsigned-to-wider-signed) is safe
      - integer-to-float is permitted without a warning
      - float-to-integer is always potentially lossy
      - f32 -> f64, f16/bf16 -> f32/f64 are safe
      - f64 -> f32/f16/bf16, f32 -> f16/bf16, f16 <-> bf16 are lossy
      - converting to or from a quantized type is lossy except
        q7/q15 -> f32, any q -> f64, and identical types
    """
    if src == dst or src is None or dst is None:
        return False
    if src[0] != 'prim' or dst[0] != 'prim':
        return False
    s, d = src[1], dst[1]

    if s in INT_INFO and d in INT_INFO:
        sbits, ssigned = INT_INFO[s]
        dbits, dsigned = INT_INFO[d]
        if ssigned == dsigned:
            return dbits < sbits
        if not ssigned and dsigned:
            return dbits <= sbits
        return True

    if s in INT_INFO and d in FLOAT_NAMES:
        return False
    if s in FLOAT_NAMES and d in INT_INFO:
        return True

    if s in FLOAT_NAMES and d in FLOAT_NAMES:
        if d == 'f64':
            return False
        if d == 'f32':
            return s == 'f64'
        return True

    if s in QUANT_NAMES and d in FLOAT_NAMES:
        if d == 'f64':
            return False
        if d == 'f32':
            return s == 'q31'
        return True
    if s in FLOAT_NAMES and d in QUANT_NAMES:
        return True
    if s in INT_INFO and d in QUANT_NAMES:
        return True
    if s in QUANT_NAMES and d in INT_INFO:
        return True
    if s in QUANT_NAMES and d in QUANT_NAMES:
        return True
    return False


def constant_conversion_is_exact(value, src, dst):
    """Whether a constant value survives an otherwise-lossy conversion."""
    if src is None or dst is None:
        return False
    if src[0] != 'prim' or dst[0] != 'prim':
        return False
    s, d = src[1], dst[1]

    if s in INT_INFO and d in INT_INFO and isinstance(value, int) \
            and not isinstance(value, bool):
        bits, signed = INT_INFO[d]
        lo = -(1 << (bits - 1)) if signed else 0
        hi = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
        return lo <= value <= hi

    if s in FLOAT_NAMES and d in INT_INFO and isinstance(value, float):
        if not math.isfinite(value) or value != int(value):
            return False
        bits, signed = INT_INFO[d]
        lo = -(1 << (bits - 1)) if signed else 0
        hi = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
        return lo <= int(value) <= hi

    if s in FLOAT_NAMES and d in FLOAT_NAMES:
        if d == 'f64':
            return True
        try:
            f = float(value)
            if not math.isfinite(f):
                return False
            if d == 'f32':
                return struct.unpack('<f', struct.pack('<f', f))[0] == f
            if d == 'f16':
                return struct.unpack('<e', struct.pack('<e', f))[0] == f
            if d == 'bf16':
                bits = int.from_bytes(struct.pack('<f', f), 'little')
                return (bits & 0xFFFF) == 0
        except (OverflowError, struct.error):
            return False
    return False


def pointer_view_is_safe(src, dst):
    """True for exact pointers and byte/void-buffer views."""
    if src == dst:
        return True
    if src is None or dst is None:
        return False
    if src[0] != 'ptr' or dst[0] != 'ptr':
        return False

    def byte_view(t):
        inner = t[1]
        if inner[0] == 'prim' and inner[1] in ('u8', 'char', 'void'):
            return True
        if is_array(inner):
            elem = array_elem(inner)
            return elem[0] == 'prim' and elem[1] in ('u8', 'char')
        return False

    return byte_view(src) or byte_view(dst)


def ts(t):
    """Render a type for diagnostics."""
    if t is None:
        return 'void'
    if t[0] == 'prim':
        return t[1]
    if t[0] == 'ptr':
        return f"ptr[{ts(t[1])}]"
    if t[0] == 'array':
        return f"[{ts(t[1])}; {t[2]}]"
    if t[0] == 'named':
        return t[1]
    return str(t)
