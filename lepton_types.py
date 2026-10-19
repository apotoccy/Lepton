"""
Lepton type system.

Everything related to primitive types, their C lowering, and the small
helpers the checker uses. Kept deliberately separate from leptonc.py so
that Phase 1's numeric work (f16, bf16, q15, SIMD vectors, per-target
capability tables) has a natural home.
"""

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
