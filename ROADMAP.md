# Lepton roadmap

## Phase 0 — Refactor for growth


- [x] Test infrastructure (`tests/`, `make test`)
- [x] Modules (`import "path.lp";` and `import foo.bar;`)
- [x] Extract type definitions to `lepton_types.py`
- [x] Source spans threaded through the pipeline
- [x] IR (`lepton_ir.py`)
- [x] AST → IR → C migration (`lepton_passes.py`, `lepton_c.py`)
- [x] Delete the AST emitter (single pipeline)
- [ ] IR: unique C names per IRBind (fixes same-name-different-type shadowing)
- [ ] IR: validate declared types in params, returns, struct fields, casts



## Phase 1 — Numeric core

Not started. Foundation is ready: new numeric types land as entries in
`lepton_types.py` (`PRIMS`, `CTYPE`, `RANK`) plus new instructions in
`lepton_ir.py`, with matching cases in `lepton_lower.py` and
`lepton_c_ir.py`.

- [x] `f16` primitive (native `_Float16`, `float` fallback), literal suffix `h`
- [x] `IRCast` instruction: numeric-to-numeric, bool/int, char/int, ptr/ptr
- [x] `bf16` primitive
- [x] `q15`, `q7`, `q31` fixed-point types
- [ ] Saturation and rounding intrinsics
- [x] SIMD vector types (`v4f32`, `v8i16`, `v16i8`)
- [x] Scalar fallback lowering for SIMD
- [x] Vector literal construction and element access
- [x] Vector arithmetic (scalar fallback)
- [x] SSE2 lowering for x86-64
- [ ] NEON lowering for ARM
- [ ] Target capability table
- [ ] Math intrinsics (`sqrt`, `rsqrt`, `exp`, `log`, `tanh`, `erf`)
- [ ] Fast approximate versions, gated on a `fast` flag
- [ ] `softmax` stdlib helper
- [ ] `std/math.lp` and `std/fixed.lp` wrapper modules
- [ ] Full test suite for the above
- [ ] Extern linkage: `extern "C"` vs other ABIs (currently always C)
- [ ] Multiple translation units: `--link foo.o` or automatic `.c` in src/



## Known limitations carried forward

- [ ] Move "stringifiable types" predicate into `lepton_types.py`
- [ ] IR: validate declared types in params, returns, struct fields, casts

## Later phases

See the roadmap document for Phases 2–6.
