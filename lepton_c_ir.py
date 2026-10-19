"""
IR -> C emission.

Mirrors the output of the AST-based `Compiler` in leptonc.py, byte for
byte, for programs the IR pipeline supports. If you change the AST
emitter's structure (preamble, runtime, prototypes, spacing), change this
one the same way or tests/test_ir.py will catch the drift.
"""
from __future__ import annotations
from lepton_ir import (
    IRModule, IRFunc, IRBlock, IRTemp, IRLocal, IRConst,
    IRBind, IRAssign, IRBinOp, IRUnOp, IRCall,
    IRConcat, IRToString, IRCast, IRMathOp, IRVecReduce, IRSplat,
    IRVecBinOp,
    IRArrayLit, IRArrayIndex, IRArraySet, IRRef, IRAlloc,
    IRStructInit, IRFieldGet,
    IRVecLit, IRVecIndex, IRVecSet,
    IRReturn, IRJump, IRCondJump,
)
from lepton_types import (CTYPE, PRIMS, is_int, is_vector,
                          is_array, array_elem, array_len,
                          array_ctype_name)
from leptonc import CompileError

from lepton_runtime import (make_preamble, RUNTIME, RUNTIME_MATH,
                             FREESTANDING_DECLS, PREAMBLE_HEAD,
                             PREAMBLE_READ_HELPERS, PREAMBLE_VEC_OPS)

# =============================================================================
#  Host runtime (emitted verbatim into every non-freestanding translation unit)
# =============================================================================



def c_string(s):
    """Emit a C string literal, safe for arbitrary bytes."""
    parts, cur = [], []
    for ch in s:
        o = ord(ch)
        if ch == '"':
            cur.append('\\"')
        elif ch == '\\':
            cur.append('\\\\')
        elif ch == '\n':
            cur.append('\\n')
        elif ch == '\t':
            cur.append('\\t')
        elif ch == '\r':
            cur.append('\\r')
        elif 32 <= o < 127:
            cur.append(ch)
        else:
            cur.append('\\%03o' % o)
            parts.append(''.join(cur))
            cur = []
    parts.append(''.join(cur))
    return '"' + '" "'.join(parts) + '"'


class CEmitter:
    def __init__(self, module, fname, freestanding=False, target=None):
        from lepton_target import get_target, Target
        self.module = module
        self.fname = fname
        self.freestanding = freestanding
        if target is None:
            target = 'scalar'
        if isinstance(target, str):
            target = get_target(target)
        elif not isinstance(target, Target):
            raise TypeError(f"target must be a string or Target, "
                            f"got {type(target).__name__}")
        self.target = target
        self.out = []
        self.indent = 0

    def emit(self, line=""):
        self.out.append('    ' * self.indent + line)

    def err(self, msg):
        raise CompileError(msg, 0, 0, self.fname)

    # -- types ---------------------------------------------------------------
    def resolve(self, t):
        if t is None:
            return ('prim', 'void')
        if t[0] == 'prim':
            if t[1] in PRIMS:
                return t
            self.err(f"unknown primitive type '{t[1]}'")
        if t[0] == 'ptr':
            return ('ptr', self.resolve(t[1]))
        if t[0] == 'array':
            return ('array', self.resolve(t[1]), t[2])
        if t[0] == 'named':
            if t[1] in self.module.structs or t[1] in self.module.enums:
                return t
            self.err(f"unknown type '{t[1]}'")
        self.err(f"bad type {t!r}")

    def ctype(self, t):
        t = self.resolve(t)
        if t[0] == 'prim':
            return CTYPE[t[1]]
        if t[0] == 'ptr':
            return self.ctype(t[1]) + '*'
        if t[0] == 'array':
            return array_ctype_name(t[1], t[2])
        if t[0] == 'named':
            return t[1]
        self.err(f"bad type {t!r}")

    # -- top level -----------------------------------------------------------
    def compile(self):
        self.emit(make_preamble(self.target))
        if self.freestanding:
            self.emit(FREESTANDING_DECLS)
        else:
            self.emit(RUNTIME)
        self.emit(RUNTIME_MATH)
        self.emit()

        for name, fields in self.module.structs.items():
            self.emit(f"typedef struct {name} {{")
            self.indent += 1
            for fname, ftype in fields:
                self.emit(f"{self.ctype(self.resolve(ftype))} {fname};")
            self.indent -= 1
            self.emit(f"}} {name};")
            self.emit()

        for name, variants in self.module.enums.items():
            self.emit("typedef enum {")
            self.indent += 1
            self.emit(", ".join(f"{name}_{v}" for v in variants))
            self.indent -= 1
            self.emit(f"}} {name};")
            self.emit()
            
        array_types = self.collect_array_types()
        if array_types:
            for elem, count in sorted(array_types,
                                      key=lambda k: (k[1], str(k[0]))):
                name = array_ctype_name(elem, count)
                self.emit(f"typedef struct {{ {self.ctype(elem)} "
                          f"e[{count}]; }} {name};")
            self.emit()

        self.emit()   # blank line the AST emitter also emits after consts
        
        for name, (params, ret) in self.module.externs.items():
            r = self.resolve(ret) if ret else ('prim', 'void')
            ps = ", ".join(f"{self.ctype(self.resolve(p[1]))} {p[0]}"
                           for p in params)
            if not ps:
                ps = "void"
            self.emit(f"{self.ctype(r)} {name}({ps});")

        for name, (cname, ty, value) in self.module.globals.items():
            self.emit(f"static const {self.ctype(ty)} {cname} = "
                      f"{self.emit_const(value)};")
        if self.module.globals:
            self.emit()

        for f in self.module.funcs:
            r = self.ret_type(f)
            ps = self.param_list(f)
            self.emit(f"{self.ctype(r)} {f.cname}({ps});")
        self.emit()

        for f in self.module.funcs:
            self.emit_fn(f)

        return '\n'.join(self.out) + '\n'

    def ret_type(self, f):
        if f.name == 'main':
            return ('prim', 'i32')
        return self.resolve(f.ret) if f.ret else ('prim', 'void')

    def param_list(self, f):
        ps = ", ".join(f"{self.ctype(self.resolve(p[1]))} {p[0]}"
                       for p in f.params)
        return ps if ps else "void"

    # -- function bodies ----------------------------------------------------- 
    
    def collect_array_types(self):
        """Return the set of (elem_type, count) pairs used anywhere."""
        seen = set()

        def visit(t):
            if t is None:
                return
            if is_array(t):
                key = (t[1], t[2])
                if key not in seen:
                    seen.add(key)
                    visit(t[1])
            elif t[0] == 'ptr':
                visit(t[1])

        for f in self.module.funcs:
            for _, pt in f.params:
                visit(pt)
            visit(f.ret)
            for b in f.blocks:
                for instr in b.instrs:
                    visit(getattr(instr, 'type', None))
                    for op in instr.operands():
                        visit(getattr(op, 'type', None))
        return seen    
    
    def emit_fn(self, f):
        r = self.ret_type(f)
        if f.name == 'main' and not f.params:
            sig = "int main(void)"
        else:
            sig = f"{self.ctype(r)} {f.cname}({self.param_list(f)})"
        self.emit(sig + " {")
        self.indent += 1

        # Hoist every named local to the top of the function. IR values are
        # function-scoped; C is block-scoped. Declaring all of them up front
        # makes a bind in one IR block visible to its siblings and to the
        # merge block, which is what the lowerer assumes.
        for kind, name, ty in self.hoisted_decls(f):
            init = self.zero_literal(ty) if kind == 'local' else ''
            if init:
                self.emit(f"{self.ctype(ty)} {name} = {init};")
            else:
                self.emit(f"{self.ctype(ty)} {name};")

        targets = self.jump_targets(f)
        for block in f.blocks:
            if block.label in targets:
                self.emit(f"lbl_{block.label}: {{")
                self.indent += 1
                for instr in block.instrs:
                    self.emit_instr(instr)
                self.indent -= 1
                self.emit("}")
            else:
                for instr in block.instrs:
                    self.emit_instr(instr)

        self.indent -= 1
        self.emit("}")
        self.emit()
        
    def hoisted_decls(self, f):
        """Every name that needs a top-of-function declaration:
        source-level IRBind names and IR temps. IR values are function-
        scoped; C is block-scoped, so all declarations are hoisted.

        Any instruction that carries a non-None `dst` gets its temp
        hoisted automatically, so new instructions need no update here.
        """
        seen_locals = set()
        seen_temps = set()
        result = []
        for block in f.blocks:
            for instr in block.instrs:
                if isinstance(instr, IRBind):
                    if instr.name not in seen_locals:
                        seen_locals.add(instr.name)
                        result.append(('local', instr.name, instr.type))
                dst = getattr(instr, 'dst', None)
                if isinstance(dst, IRTemp) and dst.id not in seen_temps:
                    seen_temps.add(dst.id)
                    result.append(('temp', f"_t{dst.id}", dst.type))
        return result


    def zero_literal(self, t):
        if t and t[0] == 'ptr':
            return '0'
        if is_vector(t):
            return '{{0}}'
        if t and t[0] == 'named':
            return '{0}'
        if is_array(t):
            return '{0}'
        if is_vector(t):
            return '{{0}}'
        if t and t[0] == 'prim' and t[1] in ('q7', 'q15', 'q31'):
            return '0'
        if t == ('prim', 'bf16'):
            return '((lp_bf16)0.0f)'
        if t and t[0] == 'prim' and t[1] in ('f16', 'f32', 'f64'):
            return '0.0'
        if t == ('prim', 'bool'):
            return 'false'
        if t and t[0] == 'prim' and t[1] == 'str':
            return '""'
        return '0'
       

    def jump_targets(self, f):
        targets = set()
        for block in f.blocks:
            for instr in block.instrs:
                if isinstance(instr, IRJump):
                    targets.add(instr.target)
                elif isinstance(instr, IRCondJump):
                    targets.add(instr.then_target)
                    targets.add(instr.else_target)
        return targets
    

    def emit_instr(self, instr):
        if isinstance(instr, IRCall):
            args = ", ".join(self.emit_value(a) for a in instr.args)
            call = f"{instr.callee}({args})"
            if instr.dst is None:
                self.emit(f"{call};")
            else:
                self.emit(f"_t{instr.dst.id} = {call};")
            return
            

        if isinstance(instr, IRReturn):
            if instr.value is None:
                self.emit("return;")
            else:
                self.emit(f"return {self.emit_value(instr.value)};")
            return

        if isinstance(instr, IRBind):
            # A placeholder bind for an aggregate (array or struct) is
            # already satisfied by the hoisted declaration's initializer.
            # Emitting the assignment here would produce `a = {0};`,
            # which is not valid C in statement position.
            if (isinstance(instr.value, IRConst)
                    and instr.value.value == 0
                    and (is_array(instr.value.type)
                         or instr.value.type[0] == 'named')):
                return
            self.emit(f"{instr.name} = {self.emit_value(instr.value)};")
            return

        if isinstance(instr, IRAssign):
            self.emit(f"{instr.name} = {self.emit_value(instr.value)};")
            return

            
        if isinstance(instr, IRUnOp):
            v = self.emit_value(instr.operand)
            vt = instr.operand.type
            
            if is_vector(vt) and vt == instr.type:
                if instr.op == '-':
                    self.emit(f"_t{instr.dst.id} = "
                              f"lp_{vt[1]}_neg({v});")
                    return
                self.err(f"IR: unary '{instr.op}' on {vt[1]} not supported")
            
            if vt in (('prim', 'q7'), ('prim', 'q15'), ('prim', 'q31')) \
                    and instr.type == vt:
                if instr.op == '-':
                    self.emit(f"_t{instr.dst.id} = "
                              f"lp_{vt[1]}_sub(({self.ctype(vt)})0, {v});")
                    return
                self.err(f"IR: unary '{instr.op}' on {vt[1]} not supported")
            if vt == ('prim', 'bf16') and instr.type == ('prim', 'bf16'):
                self.emit(f"_t{instr.dst.id} = "
                          f"lp_bf16_from_f32(({instr.op}"
                          f"lp_bf16_to_f32({v})));")
                return
            if vt in (('prim', 'bf16'), ('prim', 'q7'),
                      ('prim', 'q15'), ('prim', 'q31')):
                v = self.promote_to_f32(v, vt)
            self.emit(f"_t{instr.dst.id} = ({instr.op}{v});")
            return
            
        if isinstance(instr, IRBinOp):
            a = self.emit_value(instr.left)
            b = self.emit_value(instr.right)
            lt = instr.left.type
            rt = instr.right.type
            dst = instr.type
            
            # Vector result: dispatch to the runtime helper.
            if is_vector(dst):
                opmap = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}
                if instr.op not in opmap:
                    self.err(f"IR: '{instr.op}' not supported on {dst[1]}")
                intrinsic = self.target.intrinsics.get((instr.op, dst[1]))
                if intrinsic:
                    self.emit(f"_t{instr.dst.id} = "
                              f"(({self.ctype(dst)})"
                              f"{{ .v = {intrinsic}({a}.v, {b}.v) }});")
                else:
                    self.emit(f"_t{instr.dst.id} = "
                              f"lp_{dst[1]}_{opmap[instr.op]}({a}, {b});")
                return

            # Quantized result: both operands are the same q type (the
            # lowerer's common_type promotes anything else to f32), so we
            # dispatch to the runtime helper.
            if dst in (('prim', 'q7'), ('prim', 'q15'), ('prim', 'q31')):
                opmap = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}
                if instr.op not in opmap:
                    self.err(f"IR: '{instr.op}' not supported on {dst[1]}")
                self.emit(f"_t{instr.dst.id} = "
                          f"lp_{dst[1]}_{opmap[instr.op]}({a}, {b});")
                return

            # Comparisons: promote q operands to f32 unless both sides
            # are the same q type, in which case raw comparison is exact
            # (q values are 2's-complement and monotonic in value).
            if instr.op in ('==', '!=', '<', '<=', '>', '>='):
                q = ('prim', 'q7'), ('prim', 'q15'), ('prim', 'q31')
                if not (lt == rt and lt in q):
                    a = self.promote_to_f32(a, lt)
                    b = self.promote_to_f32(b, rt)

            # bf16 arithmetic goes through f32 on both paths.
            if dst == ('prim', 'bf16'):
                a_f32 = (a if lt not in (('prim', 'bf16'),
                                         ('prim', 'q7'), ('prim', 'q15'),
                                         ('prim', 'q31'))
                         else self.promote_to_f32(a, lt))
                b_f32 = (b if rt not in (('prim', 'bf16'),
                                         ('prim', 'q7'), ('prim', 'q15'),
                                         ('prim', 'q31'))
                         else self.promote_to_f32(b, rt))
                self.emit(f"_t{instr.dst.id} = "
                          f"lp_bf16_from_f32(({a_f32} {instr.op} {b_f32}));")
                return

            if lt in (('prim', 'bf16'), ('prim', 'q7'),
                      ('prim', 'q15'), ('prim', 'q31')):
                a = self.promote_to_f32(a, lt)
            if rt in (('prim', 'bf16'), ('prim', 'q7'),
                      ('prim', 'q15'), ('prim', 'q31')):
                b = self.promote_to_f32(b, rt)

            self.emit(f"_t{instr.dst.id} = ({a} {instr.op} {b});")
            return
               
        if isinstance(instr, IRStructInit):
            if instr.fields:
                parts = ", ".join(f".{n} = {self.emit_value(v)}"
                                  for n, v in instr.fields)
                body = parts
            else:
                body = "0"
            self.emit(f"_t{instr.dst.id} = "
                      f"(({instr.type[1]}){{ {body} }});")
            return

        if isinstance(instr, IRFieldGet):
            self.emit(f"_t{instr.dst.id} = "
                      f"({self.emit_value(instr.base)}).{instr.field};")
            return
                
        if isinstance(instr, IRMathOp):
            arg = self.emit_value(instr.args[0])
            dst_type = instr.type
            if is_vector(dst_type):
                intrinsic = self.target.intrinsics.get(
                    (instr.op, dst_type[1]))
                if intrinsic:
                    self.emit(f"_t{instr.dst.id} = "
                              f"(({self.ctype(dst_type)})"
                              f"{{ .v = {intrinsic}({arg}.v) }});")
                else:
                    self.emit(f"_t{instr.dst.id} = "
                              f"lp_{dst_type[1]}_{instr.op}({arg});")
                return
            self.emit(f"_t{instr.dst.id} = "
                      f"lp_{instr.op}_{dst_type[1]}({arg});")
            return
                
        if isinstance(instr, IRVecReduce):
            self.emit(f"_t{instr.dst.id} = "
                      f"lp_{instr.vec.type[1]}_{instr.op}"
                      f"({self.emit_value(instr.vec)});")
            return
                
        if isinstance(instr, IRSplat):
            val = self.emit_value(instr.value)
            splat = self.target.splats.get(instr.type[1])
            if splat:
                self.emit(f"_t{instr.dst.id} = "
                          f"(({self.ctype(instr.type)})"
                          f"{{ .v = {splat}({val}) }});")
            else:
                self.emit(f"_t{instr.dst.id} = "
                          f"lp_{instr.type[1]}_splat({val});")
            return
            
        if isinstance(instr, IRVecBinOp):
            op = instr.op
            vt = instr.type
            vec_name = vt[1]
            args = [self.emit_value(a) for a in instr.args]
            if op == 'vselect':
                intrinsic = None
            else:
                intrinsic = self.target.vec_binops.get((op, vec_name))
            if intrinsic:
                self.emit(f"_t{instr.dst.id} = "
                          f"(({self.ctype(vt)})"
                          f"{{ .v = {intrinsic}({args[0]}.v, {args[1]}.v) }});")
            else:
                call_args = ", ".join(args)
                self.emit(f"_t{instr.dst.id} = "
                          f"lp_{vec_name}_{op}({call_args});")
            return
            
        if isinstance(instr, IRArrayLit):
            elems = ", ".join(self.emit_value(e) for e in instr.elems)
            self.emit(f"_t{instr.dst.id} = "
                      f"(({self.ctype(instr.type)})"
                      f"{{ .e = {{ {elems} }} }});")
            return

        if isinstance(instr, IRArrayIndex):
            base = self.emit_value(instr.base)
            idx = self.emit_value(instr.index)
            bt = instr.base.type
            if bt == ('prim', 'str'):
                self.emit(f"_t{instr.dst.id} = ({base})[{idx}];")
            elif bt[0] == 'ptr' and bt[1][0] == 'array':
                self.emit(f"_t{instr.dst.id} = ({base})->e[{idx}];")
            elif bt[0] == 'ptr':
                self.emit(f"_t{instr.dst.id} = ({base})[{idx}];")
            else:
                self.emit(f"_t{instr.dst.id} = ({base}).e[{idx}];")
            return

        if isinstance(instr, IRArraySet):
            base = self.emit_value(instr.base)
            idx = self.emit_value(instr.index)
            val = self.emit_value(instr.value)
            bt = instr.base.type
            if bt[0] == 'ptr' and bt[1][0] == 'array':
                self.emit(f"({base})->e[{idx}] = {val};")
            elif bt[0] == 'ptr':
                self.emit(f"({base})[{idx}] = {val};")
            else:
                self.emit(f"({base}).e[{idx}] = {val};")
            return
            
        if isinstance(instr, IRRef):
            self.emit(f"_t{instr.dst.id} = "
                      f"&({self.emit_value(instr.value)});")
            return

        if isinstance(instr, IRAlloc):
            ct = self.ctype(instr.array_type)
            self.emit(f"_t{instr.dst.id} = "
                      f"({ct}*)lp_alloc(sizeof({ct}));")
            return

        if isinstance(instr, IRVecLit):
            elems = ", ".join(self.emit_value(el) for el in instr.elems)
            self.emit(f"_t{instr.dst.id} = "
                      f"(({self.ctype(instr.type)})"
                      f"{{ .e = {{ {elems} }} }});")
            return

        if isinstance(instr, IRVecIndex):
            self.emit(f"_t{instr.dst.id} = "
                      f"({self.emit_value(instr.vec)}).e"
                      f"[{self.emit_value(instr.index)}];")
            return
            
        if isinstance(instr, IRVecSet):
            self.emit(f"({self.emit_value(instr.vec)}).e"
                      f"[{self.emit_value(instr.index)}] = "
                      f"{self.emit_value(instr.value)};")
            return
                
        if isinstance(instr, IRCast):
            v = self.emit_value(instr.value)
            src = instr.value.type
            dst = instr.type
            q_types = (('prim', 'q7'), ('prim', 'q15'), ('prim', 'q31'))
            src_q = src in q_types
            dst_q = dst in q_types

            if dst_q and not src_q:
                self.emit(f"_t{instr.dst.id} = "
                          f"lp_{dst[1]}_from_f32((float){v});")
                return
            if src_q and not dst_q:
                self.emit(f"_t{instr.dst.id} = "
                          f"(({self.ctype(dst)})lp_{src[1]}_to_f32({v}));")
                return
            if src_q and dst_q and src != dst:
                self.emit(f"_t{instr.dst.id} = "
                          f"lp_{dst[1]}_from_f32(lp_{src[1]}_to_f32({v}));")
                return
            if dst == ('prim', 'bf16') and src != ('prim', 'bf16'):
                self.emit(f"_t{instr.dst.id} = "
                          f"lp_bf16_from_f32((float){v});")
                return
            if src == ('prim', 'bf16') and dst != ('prim', 'bf16'):
                self.emit(f"_t{instr.dst.id} = "
                          f"(({self.ctype(dst)})lp_bf16_to_f32({v}));")
                return
            self.emit(f"_t{instr.dst.id} = "
                      f"(({self.ctype(dst)}){v});")
            return
                   
        if isinstance(instr, IRJump):
            self.emit(f"goto lbl_{instr.target};")
            return

        if isinstance(instr, IRCondJump):
            c = self.emit_value(instr.cond)
            t = instr.cond.type
            if t == ('prim', 'bool'):
                cond_c = c
            elif is_int(t) or t[0] == 'ptr' or t[0] == 'named':
                cond_c = f"({c} != 0)"
            else:
                self.err(f"IR: cannot use '{t}' as a condition")
            self.emit(f"if ({cond_c}) goto lbl_{instr.then_target};")
            self.emit(f"goto lbl_{instr.else_target};")
            return
            
        if isinstance(instr, IRConcat):
            self.emit(f"_t{instr.dst.id} = "
                      f"lp_cat({self.emit_value(instr.left)}, "
                      f"{self.emit_value(instr.right)});")
            return

        if isinstance(instr, IRToString):
            self.emit(f"_t{instr.dst.id} = "
                      f"{self.emit_to_string(instr.value, instr.src_type)};")
            return

        self.err(f"IR: unsupported instruction {type(instr).__name__}")

    # -- operands ------------------------------------------------------------
    def emit_value(self, v):
        if isinstance(v, IRConst):
            return self.emit_const(v)
        if isinstance(v, IRLocal):
            return v.name
        if isinstance(v, IRTemp):
            return f"_t{v.id}"
        self.err(f"IR: unsupported value {v!r}")

    def emit_const(self, v):
        if v.type and v.type[0] == 'named':
            # Enum constant: value is the C identifier, emitted verbatim.
            return str(v.value)
        if v.type and v.type[0] == 'array':
            return f"(({self.ctype(v.type)}){{0}})"
        t = v.type[1] if v.type and v.type[0] == 'prim' else None
        if t == 'str':
            return f"({c_string(v.value)})"
        if t == 'bool':
            return 'true' if v.value else 'false'
        if t == 'char':
            return str(v.value)
        if t == 'f16':
            # Wrap in an explicit cast so the compiler truncates to f16
            # precision rather than inferring from the literal.
            return f"((lp_f16){repr(v.value)})"
        if t in ('f32', 'f64'):
            return repr(v.value)
        return str(v.value)
        
    def emit_to_string(self, value, t):
        c = self.emit_value(value)
        if t == ('prim', 'bool'):
            return f"lp_str_bool({c})"
        if t == ('prim', 'char'):
            return f"lp_str_char({c})"
        if t in (('prim', 'q7'), ('prim', 'q15'), ('prim', 'q31')):
            return f"lp_str_f64((double)lp_{t[1]}_to_f32({c}))"
        if t == ('prim', 'bf16'):
            return f"lp_str_f64((double)lp_bf16_to_f32({c}))"
        if t in (('prim', 'f16'), ('prim', 'f32'), ('prim', 'f64')):
            return f"lp_str_f64((double)({c}))"
        if t[0] == 'ptr':
            return f"lp_str_u64((uint64_t)(uintptr_t)({c}))"
        if is_int(t):
            if t[1].startswith('u'):
                return f"lp_str_u64((uint64_t)({c}))"
            return f"lp_str_i64((int64_t)({c}))"
        self.err(f"IR: cannot stringify type {t}")
        
    def promote_to_f32(self, c, t):
        """Return a C expression that yields c as a float-compatible value."""
        if t in (('prim', 'f16'), ('prim', 'f32'), ('prim', 'f64')):
            return c
        if t == ('prim', 'bf16'):
            return f"lp_bf16_to_f32({c})"
        if t in (('prim', 'q7'), ('prim', 'q15'), ('prim', 'q31')):
            return f"lp_{t[1]}_to_f32({c})"
        return c


def compile_ir(module, fname, freestanding=False, target=None):
    return CEmitter(module, fname, freestanding, target).compile()


def compile_source_ir(src, fname, freestanding=False, search_paths=None,
                      target=None):
    """Full pipeline: source -> AST -> IR -> C. Mirrors compile_source()."""
    from leptonc import lex, Parser, flatten_imports
    from lepton_lower import lower_module

    toks = lex(src, fname)
    items = Parser(toks, fname).parse_program()
    if search_paths is not None:
        items = flatten_imports(items, fname, search_paths)
    module = lower_module(items, fname)
    return compile_ir(module, fname, freestanding, target)
