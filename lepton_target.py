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
                 intrinsics, splats, vec_binops, vec_select=None,
                 extra_decls=""):
        self.name = name
        self.include_headers = include_headers
        self.vector_typedefs = vector_typedefs
        self.intrinsics = intrinsics
        self.splats = splats
        # Mapping from (op, vec_name) to an intrinsic. Only ops whose
        # SSE2 mapping is one instruction appear here; everything else
        # falls back to the scalar helper lp_<vec>_<op>.
        self.vec_binops = vec_binops
        self.vec_select = vec_select or {}
        self.extra_decls = extra_decls


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
    ('vmax2',  'v2f64'): '_mm_max_pd',
    ('vmin2',  'v2f64'): '_mm_min_pd',
    ('vcmpeq', 'v2f64'): '_mm_cmpeq_pd',
    ('vcmpne', 'v2f64'): '_mm_cmpneq_pd',
    ('vcmplt', 'v2f64'): '_mm_cmplt_pd',
    ('vcmple', 'v2f64'): '_mm_cmple_pd',
    ('vcmpgt', 'v2f64'): '_mm_cmpgt_pd',
    ('vcmpge', 'v2f64'): '_mm_cmpge_pd',

    ('vmax2',  'v4f32'): '_mm_max_ps',
    ('vmin2',  'v4f32'): '_mm_min_ps',
    ('vcmpeq', 'v4f32'): '_mm_cmpeq_ps',
    ('vcmpne', 'v4f32'): '_mm_cmpneq_ps',
    ('vcmplt', 'v4f32'): '_mm_cmplt_ps',
    ('vcmple', 'v4f32'): '_mm_cmple_ps',
    ('vcmpgt', 'v4f32'): '_mm_cmpgt_ps',
    ('vcmpge', 'v4f32'): '_mm_cmpge_ps',

    # SSE2 has direct equality/greater-than for signed integers.
    ('vcmpeq', 'v4i32'): '_mm_cmpeq_epi32',
    ('vcmpgt', 'v4i32'): '_mm_cmpgt_epi32',
    ('vcmpeq', 'v8i16'): '_mm_cmpeq_epi16',
    ('vcmpgt', 'v8i16'): '_mm_cmpgt_epi16',
    ('vcmpeq', 'v16i8'): '_mm_cmpeq_epi8',
    ('vcmpgt', 'v16i8'): '_mm_cmpgt_epi8',
}

X86_64_VEC_SELECT = {
    'v2f64': 'lp_v2f64_vselect_sse2',
    'v4f32': 'lp_v4f32_vselect_sse2',
    'v4i32': 'lp_v4i32_vselect_sse2',
    'v8i16': 'lp_v8i16_vselect_sse2',
    'v16i8': 'lp_v16i8_vselect_sse2',
}

X86_64_EXTRA_DECLS = r"""
/* SSE2 lane select. The comparison helpers produce all-ones/all-zeros
   masks, so a bitwise blend is exact for both floating and integer
   vectors. */
static inline lp_v2f64 lp_v2f64_vselect_sse2(lp_v2f64 m, lp_v2f64 a,
                                              lp_v2f64 b) {
    lp_v2f64 r;
    r.v = _mm_or_pd(_mm_and_pd(m.v, a.v), _mm_andnot_pd(m.v, b.v));
    return r;
}
static inline lp_v4f32 lp_v4f32_vselect_sse2(lp_v4f32 m, lp_v4f32 a,
                                              lp_v4f32 b) {
    lp_v4f32 r;
    r.v = _mm_or_ps(_mm_and_ps(m.v, a.v), _mm_andnot_ps(m.v, b.v));
    return r;
}
static inline lp_v4i32 lp_v4i32_vselect_sse2(lp_v4i32 m, lp_v4i32 a,
                                              lp_v4i32 b) {
    lp_v4i32 r;
    r.v = _mm_or_si128(_mm_and_si128(m.v, a.v),
                       _mm_andnot_si128(m.v, b.v));
    return r;
}
static inline lp_v8i16 lp_v8i16_vselect_sse2(lp_v8i16 m, lp_v8i16 a,
                                              lp_v8i16 b) {
    lp_v8i16 r;
    r.v = _mm_or_si128(_mm_and_si128(m.v, a.v),
                       _mm_andnot_si128(m.v, b.v));
    return r;
}
static inline lp_v16i8 lp_v16i8_vselect_sse2(lp_v16i8 m, lp_v16i8 a,
                                              lp_v16i8 b) {
    lp_v16i8 r;
    r.v = _mm_or_si128(_mm_and_si128(m.v, a.v),
                       _mm_andnot_si128(m.v, b.v));
    return r;
}
"""

TARGETS = {
    'scalar': Target('scalar', [], SCALAR_VEC_TYPEDEFS, {}, {}, {}),
    'x86_64': Target('x86_64', ['#include <emmintrin.h>'],
                     X86_64_VEC_TYPEDEFS, X86_64_INTRINSICS,
                     X86_64_SPLATS, X86_64_VEC_BINOPS,
                     X86_64_VEC_SELECT, X86_64_EXTRA_DECLS),
}


def get_target(name):
    if name not in TARGETS:
        raise ValueError(f"unknown target '{name}'; "
                         f"choose from {sorted(TARGETS)}")
    return TARGETS[name]
