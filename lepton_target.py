"""
Lepton target descriptions.

A Target describes what the C back end may assume about the machine the
generated C will be compiled for. The `scalar` target assumes nothing
beyond C11 and uses helper functions for every vector operation. The
`x86_64` target assumes SSE2 is available and emits intrinsics where
possible, falling back to the scalar helpers for operations SSE2 lacks
(notably 32-bit and 8-bit integer multiply).

Adding a target means adding one entry to TARGETS. The emitter consults
`intrinsics` for each vector operation; a missing key falls through to
the scalar helper, which is always emitted.
"""

from __future__ import annotations


class Target:
    def __init__(self, name, include_headers, vector_typedefs,
                 intrinsics, splats, vec_binops):
        self.name = name
        self.include_headers = include_headers
        self.vector_typedefs = vector_typedefs
        self.intrinsics = intrinsics
        self.splats = splats
        # Mapping from (op, vec_name) to an intrinsic. Only ops whose
        # SSE2 mapping is one instruction appear here; everything else
        # falls back to the scalar helper lp_<vec>_<op>.
        self.vec_binops = vec_binops


SCALAR_VEC_TYPEDEFS = {
    'v2f64': 'typedef struct { double  e[2];  } lp_v2f64;',
    'v4f32': 'typedef struct { float   e[4];  } lp_v4f32;',
    'v4i32': 'typedef struct { int32_t e[4];  } lp_v4i32;',
    'v8i16': 'typedef struct { int16_t e[8];  } lp_v8i16;',
    'v16i8': 'typedef struct { int8_t  e[16]; } lp_v16i8;',
}

X86_64_VEC_TYPEDEFS = {
    'v2f64': 'typedef union { __m128d v; double  e[2];  } lp_v2f64;',
    'v4f32': 'typedef union { __m128  v; float   e[4];  } lp_v4f32;',
    'v4i32': 'typedef union { __m128i v; int32_t e[4];  } lp_v4i32;',
    'v8i16': 'typedef union { __m128i v; int16_t e[8];  } lp_v8i16;',
    'v16i8': 'typedef union { __m128i v; int8_t  e[16]; } lp_v16i8;',
}

# Ops SSE2 provides directly. Everything else falls back to scalar.
X86_64_INTRINSICS = {
    ('+', 'v2f64'): '_mm_add_pd',
    ('-', 'v2f64'): '_mm_sub_pd',
    ('*', 'v2f64'): '_mm_mul_pd',
    ('/', 'v2f64'): '_mm_div_pd',
    ('+', 'v4f32'): '_mm_add_ps',
    ('-', 'v4f32'): '_mm_sub_ps',
    ('*', 'v4f32'): '_mm_mul_ps',
    ('/', 'v4f32'): '_mm_div_ps',
    ('+', 'v4i32'): '_mm_add_epi32',
    ('-', 'v4i32'): '_mm_sub_epi32',
    ('+', 'v8i16'): '_mm_add_epi16',
    ('-', 'v8i16'): '_mm_sub_epi16',
    ('*', 'v8i16'): '_mm_mullo_epi16',
    ('+', 'v16i8'): '_mm_add_epi8',
    ('-', 'v16i8'): '_mm_sub_epi8',
    # Vector math. SSE2 provides sqrt and rsqrt directly; exp, tanh,
    # and erf are not in SSE2 and fall back to the scalar helpers.
    ('sqrt',  'v4f32'): '_mm_sqrt_ps',
    ('rsqrt', 'v4f32'): '_mm_rsqrt_ps',
}

X86_64_SPLATS = {
    'v2f64': '_mm_set1_pd',
    'v4f32': '_mm_set1_ps',
    'v4i32': '_mm_set1_epi32',
    'v8i16': '_mm_set1_epi16',
    'v16i8': '_mm_set1_epi8',
}

X86_64_VEC_BINOPS = {
    ('vmax2',  'v4f32'): '_mm_max_ps',
    ('vmin2',  'v4f32'): '_mm_min_ps',
    ('vcmplt', 'v4f32'): '_mm_cmplt_ps',
    ('vcmpgt', 'v4f32'): '_mm_cmpgt_ps',
}

TARGETS = {
    'scalar': Target('scalar', [], SCALAR_VEC_TYPEDEFS, {}, {}, {}),
    'x86_64': Target('x86_64', ['#include <emmintrin.h>'],
                     X86_64_VEC_TYPEDEFS, X86_64_INTRINSICS,
                     X86_64_SPLATS, X86_64_VEC_BINOPS),
}


def get_target(name):
    if name not in TARGETS:
        raise ValueError(f"unknown target '{name}'; "
                         f"choose from {sorted(TARGETS)}")
    return TARGETS[name]
