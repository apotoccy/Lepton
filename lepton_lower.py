"""
AST -> IR lowering.

Only a small subset of the language is handled here yet. Everything else
raises `IRNotImplemented`, which the test harness reports as a skip. That
is deliberate: the goal is a *correct* pipeline for one construct, not a
speculative one for thirty.

Adding a feature:
  1. Add a case to `lower_stmt` or `lower_expr`.
  2. Add a matching case to `CEmitter.emit_instr` in lepton_c_ir.py.
  3. Add or extend a test case in tests/cases/.
"""

from __future__ import annotations

from lepton_ir import (
    IRModule, IRFunc, IRLocal, IRConst,
    IRBind, IRAssign, IRBinOp, IRUnOp, IRCall,
    IRConcat, IRToString, IRCast, IRMathOp, IRVecReduce, IRSplat,
    IRVecBinOp, IRArrayLit, IRArrayIndex, IRArraySet, IRRef, IRAlloc,
    IRStructInit, IRFieldGet, IRFieldSet,
    IRVecLit, IRVecIndex, IRVecSet,
    IRReturn, IRJump, IRCondJump, IRNotImplemented,
)

from leptonc import CompileError, CompileWarning
from lepton_types import (is_num, RANK, PRIMS, ts,
                          is_vector, vector_elem, vector_len, vector_of,
                          is_array, array_elem, array_len,
                          is_lossy_conversion, pointer_view_is_safe,
                          constant_conversion_is_exact)


# Source-level builtin name -> C-level name.
BUILTIN_C_NAMES = {
    'print':   'lp_print',
    'println': 'lp_println',
    'panic':   'lp_panic',
}

# Source-level math intrinsics. These lower to IRMathOp, not to a C call.
MATH_INTRINSICS = {'sqrt', 'rsqrt', 'exp', 'tanh', 'erf', 'cos', 'sin'}

# File I/O builtins. Each is a thin wrapper around a runtime helper.
FILE_BUILTINS = {'file_size', 'read_file_at', 'write_file'}

# Little-endian read builtins. Signature: (buf: ptr[T], offset: int)
# Return type and C helper name per builtin.
READ_BUILTINS = {
    'read_u8':  ('u8',  'lp_rd_u8'),
    'read_u16': ('u16', 'lp_rd_u16_le'),
    'read_u32': ('u32', 'lp_rd_u32_le'),
    'read_u64': ('u64', 'lp_rd_u64_le'),
    'read_i8':  ('i8',  'lp_rd_i8_le'),
    'read_i32': ('i32', 'lp_rd_i32_le'),
    'read_i64': ('i64', 'lp_rd_i64_le'),
    'read_f16': ('f32', 'lp_rd_f16_le'),
    'read_f32': ('f32', 'lp_rd_f32_le'),
}

STRING_BUILTINS = {'read_str'}

# Reductions over a vector. Lower to IRVecReduce, not to a C call.
VEC_REDUCTIONS = {'sum': 'sum', 'vmax': 'max', 'vmin': 'min'}

# Vector builtins whose argument shape is not the usual one-arg-reduce
# or one-arg-math pattern. Each entry maps a source name to (op, arity)
# where op is the IRVecBinOp op and arity is how many positional args
# the source form takes.
VEC_BIN_OPS = {
    'vmax2':   ('vmax2',   2),
    'vmin2':   ('vmin2',   2),
    'vcmpeq':  ('vcmpeq',  2),
    'vcmpne':  ('vcmpne',  2),
    'vcmplt':  ('vcmplt',  2),
    'vcmple':  ('vcmple',  2),
    'vcmpgt':  ('vcmpgt',  2),
    'vcmpge':  ('vcmpge',  2),
    'vselect': ('vselect', 3),
}

class LValue:
    """A lowered assignment target.

    `read()` emits/returns the current value; `write(value)` emits a store.
    `direct_local` means the value is an IRLocal that can be mutated in
    place without copying and writing back through a parent.
    """

    def __init__(self, type, read, write, mutable=True,
                 direct_local=False, desc="value"):
        self.type = type
        self.read = read
        self.write = write
        self.mutable = mutable
        self.direct_local = direct_local
        self.desc = desc


class Lowering:
    def __init__(self, items, fname, strict_types=False):
        self.items = items
        self.fname = fname
        self.strict_types = strict_types
        self.warnings = []
        self.module = IRModule()
        self.func = None
        self.block = None
        self.scopes = []
        self.loop_stack = []
        self.declared_names = set()
        self.mutable_names = set()
        self.rename_counter = 0
        self.current_ret = None
        self.main_implicit_return = False

    # -- diagnostics ---------------------------------------------------------
    def err(self, msg, pos=None):
        if pos is None:
            raise CompileError(msg, 0, 0, self.fname)
        raise CompileError(msg, pos[0], pos[1], self.fname)

    def not_impl(self, msg):
        raise IRNotImplemented(msg)

    def warn(self, msg, pos=None):
        line, col = pos if pos else (0, 0)
        self.warnings.append(CompileWarning(msg, line, col, self.fname))

    def conversion_warning(self, src, dst, pos, context, kind):
        msg = (f"implicit {kind} conversion from '{ts(src)}' to "
               f"'{ts(dst)}' in {context}")
        if self.strict_types:
            self.err(f"{msg}; use an explicit 'as' cast", pos)
        self.warn(f"{msg}; use 'as' for an explicit conversion", pos)

    # -- driver --------------------------------------------------------------
    def lower(self):
        # Pass 1a: nominal types first, so resolve_type works below.
        for it in self.items:
            if it.kind == 'struct':
                self.module.structs[it.name] = list(it.fields)
            elif it.kind == 'enum':
                self.module.enums[it.name] = list(it.variants)

        # Pass 1b: functions, externs, and constants.
        for it in self.items:
            if it.kind == 'fn':
                cname = 'main' if it.name == 'main' else 'lp_' + it.name
                params = []
                for p in it.params:
                    pt = self.resolve_type(p.type, p.pos)
                    if pt == ('prim', 'void'):
                        self.err(f"parameter '{p.name}' cannot have type "
                                 f"'void'", p.pos)
                    params.append((p.name, pt))
                ret = self.resolve_type(it.ret, it.pos) if it.ret else None
                f = IRFunc(it.name, params, ret, it.pos)
                f.cname = cname
                self.module.funcs.append(f)
            elif it.kind == 'extern':
                params = []
                for p in it.params:
                    pt = self.resolve_type(p.type, p.pos)
                    if pt == ('prim', 'void'):
                        self.err(f"parameter '{p.name}' cannot have type "
                                 f"'void'", p.pos)
                    params.append((p.name, pt))
                ret = self.resolve_type(it.ret, it.pos) if it.ret else None
                self.module.externs[it.name] = (params, ret)
            elif it.kind == 'const':
                ty = self.resolve_type(it.type, it.pos)
                val = self.eval_const(it.value, ty)
                self.module.globals[it.name] = (it.name, ty, val)
            elif it.kind == 'import':
                self.not_impl("IR: imports must be flattened before lowering")

        # Pass 2: lower function bodies.
        for it in self.items:
            if it.kind == 'fn':
                self.lower_fn(it, self.func_by_name(it.name))

        return self.module

    def func_by_name(self, name):
        for f in self.module.funcs:
            if f.name == name:
                return f
        return None

    # -- scopes --------------------------------------------------------------
    def push_scope(self):
        self.scopes.append({})
        
    def resolve_type(self, t, pos):
        """Validate a type reference against the module's known types.
        Returns the type unchanged on success."""
        if t[0] == 'prim':
            if t[1] in PRIMS:
                return t
            self.err(f"unknown type '{t[1]}'", pos)
        if t[0] == 'ptr':
            return ('ptr', self.resolve_type(t[1], pos))
        if t[0] == 'array':
            return ('array', self.resolve_type(t[1], pos), t[2])
        if t[0] == 'named':
            if t[1] in self.module.structs or t[1] in self.module.enums:
                return t
            self.err(f"unknown type '{t[1]}'", pos)
        self.err(f"bad type {t!r}", pos)

    def eval_const(self, node, target_type):
        """Evaluate a constant expression at collect time. Supports
        literals and casts of literals. Returns an IRConst whose type is
        `target_type`. Const initializers that reference other consts or
        call functions are not supported yet."""
        if node.kind == 'int':
            return IRConst(node.value, target_type or ('prim', 'i32'))
        if node.kind == 'float':
            return IRConst(node.value, target_type or ('prim', 'f64'))
        if node.kind == 'float32':
            return IRConst(node.value, target_type or ('prim', 'f32'))
        if node.kind == 'float16':
            return IRConst(node.value, target_type or ('prim', 'f16'))
        if node.kind == 'bool':
            return IRConst(node.value, target_type or ('prim', 'bool'))
        if node.kind == 'charlit':
            return IRConst(node.value, target_type or ('prim', 'char'))
        if node.kind == 'cast':
            inner = self.eval_const(node.expr, None)
            return IRConst(inner.value, target_type)
        self.err("const initializer must be a literal or a cast of a literal",
                 node.pos)

    def pop_scope(self):
        self.scopes.pop()

    def declare(self, name, type, mutable=True):
        """Declare `name` in the current scope. Returns the C-level name
        (equal to `name` if no shadowing occurred, otherwise `name_N`)."""
        cname = name
        if name in self.declared_names:
            while True:
                self.rename_counter += 1
                cname = f"{name}_{self.rename_counter}"
                if cname not in self.declared_names:
                    break
        self.declared_names.add(cname)
        if mutable:
            self.mutable_names.add(cname)
        self.scopes[-1][name] = (cname, type)
        return cname

    def lookup(self, name):
        """Return (cname, type) for the innermost binding of `name`,
        or None if not found."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    # -- functions -----------------------------------------------------------
    def lower_fn(self, it, f):
        self.func = f
        self.declared_names = set()
        self.mutable_names = set()
        self.rename_counter = 0
        self.scopes = [{}]
        self.loop_stack = []

        if f.ret is not None:
            f.ret = self.resolve_type(f.ret, it.pos)

        resolved_params = []
        for pname, ptype in f.params:
            rp = self.resolve_type(ptype, it.pos)
            if rp == ('prim', 'void'):
                self.err(f"parameter '{pname}' cannot have type 'void'",
                         it.pos)
            resolved_params.append((pname, rp))
        f.params = resolved_params

        if f.ret is None:
            self.current_ret = (('prim', 'i32') if it.name == 'main'
                                else ('prim', 'void'))
        else:
            self.current_ret = f.ret
        self.main_implicit_return = (it.name == 'main' and f.ret is None)

        for pname, ptype in f.params:
            # Params keep their source names but count toward declared_names
            # so a `let` in the body that shadows them gets renamed.
            self.scopes[0][pname] = (pname, ptype)
            self.declared_names.add(pname)
            self.mutable_names.add(pname)

        self.block = f.add_block('entry', it.pos)
        for s in it.body.stmts:
            self.lower_stmt(s)

        if not self.block.is_terminated():
            # A merge block after an if/else where both branches return is
            # unreachable; do not demand a return from it.
            if self.block.label in self.reachable_blocks(f):
                if it.name == 'main':
                    self.block.add(IRReturn(IRConst(0, ('prim', 'i32')),
                                            it.pos))
                elif f.ret is None or f.ret == ('prim', 'void'):
                    self.block.add(IRReturn(None, it.pos))
                else:
                    self.err(f"not all paths in function '{it.name}' return "
                             f"a value of type '{ts(f.ret)}'", it.pos)

    def reachable_blocks(self, f):
        """Labels reachable from the entry block. Used only to avoid
        complaining about an unreachable fallthrough merge block."""
        by_label = {b.label: b for b in f.blocks}
        seen = set()
        stack = [f.entry]
        while stack:
            b = stack.pop()
            if b is None or b.label in seen:
                continue
            seen.add(b.label)
            for instr in b.instrs:
                if isinstance(instr, IRReturn):
                    break
                if isinstance(instr, IRJump):
                    stack.append(by_label.get(instr.target))
                    break
                if isinstance(instr, IRCondJump):
                    stack.append(by_label.get(instr.then_target))
                    stack.append(by_label.get(instr.else_target))
                    break
        return seen

    # -- lvalues -------------------------------------------------------------
    def copy_to_local(self, value, ty, pos):
        """Copy a value into a fresh local so a nested lvalue can be updated."""
        name = self.func.new_label("lv")
        self.block.add(IRBind(name, value, ty, True, pos))
        return IRLocal(name, ty)

    def require_mutable(self, lv, pos):
        if not lv.mutable:
            self.err(f"cannot assign to immutable {lv.desc}", pos)

    def lower_lvalue(self, e, pos):
        """Lower an assignment target to a read/write LValue.

        Index and field bases are evaluated once. Nested aggregates are
        copied, updated, and written back through their parent.
        """
        if e.kind == 'ident':
            r = self.lookup(e.name)
            if r is None:
                self.err(f"undefined name '{e.name}'", e.pos)
            cname, ty = r
            return LValue(
                ty,
                lambda: IRLocal(cname, ty),
                lambda v: self.block.add(IRAssign(cname, v, pos)),
                cname in self.mutable_names,
                True,
                f"'{e.name}'")

        if e.kind == 'field':
            base = self.lower_lvalue(e.base, e.pos)
            bt = base.type
            if bt[0] != 'named' or bt[1] not in self.module.structs:
                self.err(f"cannot access field '{e.name}' on "
                         f"'{ts(bt)}'", e.pos)
            ftype = None
            for fname, declared in self.module.structs[bt[1]]:
                if fname == e.name:
                    ftype = declared
                    break
            if ftype is None:
                self.err(f"struct '{bt[1]}' has no field '{e.name}'", e.pos)

            def read_field():
                bv = base.read()
                tmp = self.func.new_temp(ftype)
                self.block.add(IRFieldGet(tmp, bv, e.name, ftype, e.pos))
                return tmp

            def write_field(v):
                bv = base.read()
                if base.direct_local:
                    self.block.add(IRFieldSet(bv, e.name, v, e.pos))
                else:
                    copy = self.copy_to_local(bv, base.type, e.pos)
                    self.block.add(IRFieldSet(copy, e.name, v, e.pos))
                    base.write(copy)

            return LValue(ftype, read_field, write_field, base.mutable,
                          False, f"field '{e.name}'")

        if e.kind == 'index':
            if e.base.kind not in ('ident', 'field', 'index'):
                self.not_impl("IR: indexed assignment target must be a "
                              "variable, field, or index")
            base = self.lower_lvalue(e.base, e.pos)
            idx = self.lower_expr(e.index)
            if idx is None:
                self.err("index expression has no value", e.pos)
            if not is_int(idx.type):
                self.err(f"index must be an integer, got '{ts(idx.type)}'",
                         e.pos)
            bt = base.type

            if bt[0] == 'ptr':
                elem = self.index_element_type(bt)
                if elem is None:
                    self.not_impl(f"IR: indexed assignment to '{ts(bt)}'")

                def read_ptr():
                    bv = base.read()
                    tmp = self.func.new_temp(elem)
                    self.block.add(IRArrayIndex(tmp, bv, idx, elem, e.pos))
                    return tmp

                def write_ptr(v):
                    self.block.add(IRArraySet(base.read(), idx, v, e.pos))
                return LValue(elem, read_ptr, write_ptr, True, False,
                              "pointer element")

            if is_array(bt):
                elem = array_elem(bt)

                def read_arr():
                    bv = base.read()
                    tmp = self.func.new_temp(elem)
                    self.block.add(IRArrayIndex(tmp, bv, idx, elem, e.pos))
                    return tmp

                def write_arr(v):
                    bv = base.read()
                    if base.direct_local:
                        self.block.add(IRArraySet(bv, idx, v, e.pos))
                    else:
                        copy = self.copy_to_local(bv, bt, e.pos)
                        self.block.add(IRArraySet(copy, idx, v, e.pos))
                        base.write(copy)
                return LValue(elem, read_arr, write_arr, base.mutable,
                              False, "array element")

            if is_vector(bt):
                elem = vector_elem(bt)

                def read_vec():
                    bv = base.read()
                    tmp = self.func.new_temp(elem)
                    self.block.add(IRVecIndex(tmp, bv, idx, elem, e.pos))
                    return tmp

                def write_vec(v):
                    bv = base.read()
                    if base.direct_local:
                        self.block.add(IRVecSet(bv, idx, v, e.pos))
                    else:
                        copy = self.copy_to_local(bv, bt, e.pos)
                        self.block.add(IRVecSet(copy, idx, v, e.pos))
                        base.write(copy)
                return LValue(elem, read_vec, write_vec, base.mutable,
                              False, "vector element")

            self.not_impl(f"IR: indexed assignment to '{ts(bt)}'")

        self.not_impl(f"IR: assignment target '{e.kind}'")

    # -- statements ----------------------------------------------------------
    def lower_stmt(self, s):
        if self.block.is_terminated():
            return
        k = s.kind

        if k == 'exprstmt':
            self.lower_expr(s.expr)
            return

        if k == 'return':
            if self.main_implicit_return:
                if s.value is None:
                    self.block.add(IRReturn(
                        IRConst(0, ('prim', 'i32')), s.pos))
                else:
                    v = self.lower_expr_as(s.value, ('prim', 'i32'),
                                           s.value.pos, "return value")
                    self.block.add(IRReturn(v, s.pos))
                return
            rt = self.current_ret
            if rt is None or rt == ('prim', 'void'):
                if s.value is not None:
                    self.err("cannot return a value from a void function",
                             s.pos)
                self.block.add(IRReturn(None, s.pos))
                return
            if s.value is None:
                self.err(f"function '{self.func.name}' must return a value "
                         f"of type '{ts(rt)}'", s.pos)
            v = self.lower_expr_as(s.value, rt, s.value.pos, "return value")
            self.block.add(IRReturn(v, s.pos))
            return
            
        if k == 'block':
            self.push_scope()
            for sub in s.stmts:
                self.lower_stmt(sub)
            self.pop_scope()
            return

        if k == 'let':
            declared = (self.resolve_type(s.type, s.pos)
                        if s.type is not None else None)
            if declared == ('prim', 'void'):
                self.err(f"cannot declare '{s.name}' with type 'void'", s.pos)

            if s.value is None:
                if declared is None:
                    self.err("'let' needs a type or an initialiser", s.pos)
                cname = self.declare(s.name, declared, s.mutable)
                self.block.add(IRBind(cname, self.zero_const(declared),
                                      declared, s.mutable, s.pos))
                return

            if declared is None:
                v = self.lower_expr(s.value)
                if v is None:
                    self.err("cannot bind a void expression to a name", s.pos)
                cname = self.declare(s.name, v.type, s.mutable)
                self.block.add(IRBind(cname, v, v.type, s.mutable, s.pos))
                return

            v = self.lower_expr_as(s.value, declared, s.value.pos,
                                   f"initializer for '{s.name}'")
            cname = self.declare(s.name, declared, s.mutable)
            self.block.add(IRBind(cname, v, declared, s.mutable, s.pos))
            return

        if k == 'assign':
            lv = self.lower_lvalue(s.target, s.pos)
            self.require_mutable(lv, s.pos)

            if s.op == '=':
                v = self.lower_expr_as(s.value, lv.type, s.value.pos,
                                       f"assignment to {lv.desc}")
                lv.write(v)
                return

            rhs = self.lower_expr(s.value)
            if rhs is None:
                self.err("compound assignment needs a value", s.pos)
            current = lv.read()
            result = self.lower_binop_values(current, rhs, s.op[:-1], s.pos)
            result = self.coerce(result, lv.type, s.value.pos,
                                 f"compound assignment to {lv.desc}")
            lv.write(result)
            return

        if k == 'if':
            self.lower_if(s)
            return

        if k == 'while':
            self.lower_while(s)
            return

            
        if k == 'for':
            self.lower_for(s)
            return

        if k == 'break':
            if not self.loop_stack:
                self.err("'break' outside a loop", s.pos)
            _, end_lbl = self.loop_stack[-1]
            self.block.add(IRJump(end_lbl, s.pos))
            return

        if k == 'continue':
            if not self.loop_stack:
                self.err("'continue' outside a loop", s.pos)
            cont_lbl, _ = self.loop_stack[-1]
            self.block.add(IRJump(cont_lbl, s.pos))
            return

        if k == 'raw':
            self.not_impl("IR: 'raw' blocks not yet lowered")

        self.not_impl(f"IR: statement '{k}' not yet lowered")

    # -- type coercion -------------------------------------------------------
    def zero_const(self, t):
        """A zero value of type `t` for an uninitialised declaration."""
        if t == ('prim', 'str'):
            return IRConst("", t)
        if t == ('prim', 'bool'):
            return IRConst(False, t)
        return IRConst(0, t)

    def index_element_type(self, t):
        """Element type produced by indexing a value of type `t`."""
        if is_array(t):
            return array_elem(t)
        if t[0] == 'ptr' and is_array(t[1]):
            return array_elem(t[1])
        if t[0] == 'ptr' and t[1][0] in ('prim', 'ptr', 'named'):
            return t[1]
        if is_vector(t):
            return vector_elem(t)
        return None

    def lower_expr_as(self, e, target, pos, context):
        """Lower `e` and make it assignable to `target`.

        Array and vector literals are lowered directly against the target
        type, so `[f32; 4]` gets f32 elements rather than the literal's
        inferred common type. Everything else is lowered normally and then
        coerced.
        """
        if target is None:
            return self.lower_expr(e)
        if target == ('prim', 'void'):
            self.err(f"{context} cannot have type 'void'", pos or e.pos)
        if is_array(target) and e.kind == 'arrlit':
            return self.lower_arrlit_as(e, target, pos)
        if is_vector(target) and e.kind == 'veclit':
            return self.lower_veclit_as(e, target, pos)
        v = self.lower_expr(e)
        if v is None:
            self.err(f"{context} has no value", pos or e.pos)
        return self.coerce(v, target, pos or e.pos, context)

    def coerce(self, v, target, pos, context):
        """Return `v` converted to `target`, or raise a type error.

        Numeric values may be converted implicitly because the destination
        type is explicit. Pointers may be cast through one another, which is
        needed for byte-buffer views. Everything else must match exactly.
        """
        if v.type == target:
            return v
        if target == ('prim', 'void'):
            self.err(f"{context} cannot have type 'void'", pos)
        if is_num(v.type) and is_num(target):
            if is_lossy_conversion(v.type, target):
                exact_const = (isinstance(v, IRConst)
                               and constant_conversion_is_exact(
                                   v.value, v.type, target))
                if not exact_const:
                    self.conversion_warning(v.type, target, pos, context,
                                            "lossy numeric")
            tmp = self.func.new_temp(target)
            self.block.add(IRCast(tmp, v, target, pos))
            return tmp
        if v.type[0] == 'ptr' and target[0] == 'ptr':
            if not pointer_view_is_safe(v.type, target):
                self.conversion_warning(v.type, target, pos, context,
                                        "unsafe pointer")
            tmp = self.func.new_temp(target)
            self.block.add(IRCast(tmp, v, target, pos))
            return tmp
        if (target[0] == 'ptr' and isinstance(v, IRConst)
                and is_int(v.type) and v.value == 0):
            tmp = self.func.new_temp(target)
            self.block.add(IRCast(tmp, v, target, pos))
            return tmp
        self.err(f"cannot assign '{ts(v.type)}' to '{ts(target)}' "
                 f"in {context}", pos)

    # -- expressions ---------------------------------------------------------
    def lower_condition(self, e, what):
        v = self.lower_expr(e)
        if v is None:
            self.err(f"{what} condition has no value", e.pos)
        if v.type != ('prim', 'bool'):
            self.err(f"{what} condition must be bool, got '{ts(v.type)}'; "
                     f"use a comparison", e.pos)
        return v

    def check_comparable(self, l, r, op, pos):
        lt, rt = l.type, r.type
        if is_vector(lt) or is_vector(rt):
            self.err("vector comparison not supported; use element access "
                     "or a reduction", pos)
        if is_array(lt) or is_array(rt):
            self.err(f"cannot compare arrays '{ts(lt)}' and '{ts(rt)}'", pos)
        if lt == ('prim', 'bool') or rt == ('prim', 'bool'):
            if lt != rt:
                self.err(f"cannot compare '{ts(lt)}' and '{ts(rt)}'", pos)
            if op not in ('==', '!='):
                self.err("bool values only support '==' and '!='", pos)
            return
        if lt[0] == 'named' or rt[0] == 'named':
            if lt != rt:
                self.err(f"cannot compare '{ts(lt)}' and '{ts(rt)}'", pos)
            if lt[1] in self.module.structs:
                self.err(f"cannot compare structs; compare fields of "
                         f"'{lt[1]}' explicitly", pos)
            if lt[1] in self.module.enums:
                return
            self.err(f"cannot compare '{ts(lt)}'", pos)
        if lt[0] == 'ptr' or rt[0] == 'ptr':
            if lt != rt:
                self.err(f"cannot compare different pointer types "
                         f"'{ts(lt)}' and '{ts(rt)}'", pos)
            return
        if is_num(lt) and is_num(rt):
            return
        self.err(f"cannot compare '{ts(lt)}' and '{ts(rt)}'", pos)

    def lower_binop_values(self, l, r, op, pos):
        # Pointer arithmetic: ptr ± int, int + ptr, ptr - ptr.
        if op in ('+', '-') and (l.type[0] == 'ptr' or r.type[0] == 'ptr'):
            return self.lower_ptr_arith_values(l, r, op, pos)

        if op in ('==', '!=', '<', '<=', '>', '>='):
            if l.type == ('prim', 'str') or r.type == ('prim', 'str'):
                if l.type != ('prim', 'str') or r.type != ('prim', 'str'):
                    self.err(f"cannot compare '{ts(l.type)}' and "
                             f"'{ts(r.type)}'", pos)
                return self.lower_str_compare(l, r, op, pos)
            self.check_comparable(l, r, op, pos)
            t = ('prim', 'bool')
        else:
            # Broadcast a scalar operand to match a vector operand.
            if is_vector(l.type) and not is_vector(r.type):
                r = self.broadcast(r, l.type, pos)
            elif is_vector(r.type) and not is_vector(l.type):
                l = self.broadcast(l, r.type, pos)
            t = self.common_type(l.type, r.type, pos)
        dst = self.func.new_temp(t)
        self.block.add(IRBinOp(dst, op, l, r, t, pos))
        return dst

    def lower_str_compare(self, l, r, op, pos):
        if op in ('==', '!='):
            eq = self.func.new_temp(('prim', 'bool'))
            self.block.add(IRCall(eq, 'lp_str_eq', [l, r],
                                  ('prim', 'bool'), pos))
            if op == '==':
                return eq
            ne = self.func.new_temp(('prim', 'bool'))
            self.block.add(IRUnOp(ne, '!', eq, ('prim', 'bool'), pos))
            return ne

        cmp = self.func.new_temp(('prim', 'i32'))
        self.block.add(IRCall(cmp, 'lp_str_cmp', [l, r],
                              ('prim', 'i32'), pos))
        zero = IRConst(0, ('prim', 'i32'))
        dst = self.func.new_temp(('prim', 'bool'))
        self.block.add(IRBinOp(dst, op, cmp, zero, ('prim', 'bool'), pos))
        return dst

    def lower_logical(self, e, is_and):
        l = self.lower_expr(e.left)
        if l is None:
            self.err(f"'{e.op}' needs value operands", e.pos)
        if l.type != ('prim', 'bool'):
            self.err(f"'{e.op}' operands must be bool, got '{ts(l.type)}'",
                     e.pos)

        # A fresh C-visible local to hold the result. The name comes from
        # the label counter so it is unique per function.
        dst_name = self.func.new_label("sc")
        placeholder = IRConst(False if is_and else True, ('prim', 'bool'))
        self.block.add(IRBind(dst_name, placeholder, ('prim', 'bool'),
                              True, e.pos))

        rhs_lbl   = self.func.new_label("rhs")
        short_lbl = self.func.new_label("short")
        merge_lbl = self.func.new_label("merge")

        # For &&: if left then evaluate right, else short-circuit to false.
        # For ||: if left then short-circuit to true, else evaluate right.
        if is_and:
            self.block.add(IRCondJump(l, rhs_lbl, short_lbl, e.pos))
        else:
            self.block.add(IRCondJump(l, short_lbl, rhs_lbl, e.pos))

        # Right operand.
        self.block = self.func.add_block(rhs_lbl, e.pos)
        r = self.lower_expr(e.right)
        if r is None:
            self.err(f"'{e.op}' needs value operands", e.pos)
        if r.type != ('prim', 'bool'):
            self.err(f"'{e.op}' operands must be bool, got '{ts(r.type)}'",
                     e.pos)
        self.block.add(IRAssign(dst_name, r, e.pos))
        self.block.add(IRJump(merge_lbl, e.pos))

        # Short-circuit path: dst retains its placeholder value.
        self.block = self.func.add_block(short_lbl, e.pos)
        self.block.add(IRJump(merge_lbl, e.pos))

        self.block = self.func.add_block(merge_lbl, e.pos)
        return IRLocal(dst_name, ('prim', 'bool'))
    
    def lower_if(self, s):
        cond = self.lower_condition(s.cond, "if")

        then_lbl  = self.func.new_label("then")
        merge_lbl = self.func.new_label("merge")
        else_lbl  = self.func.new_label("else") if s.els else merge_lbl

        self.block.add(IRCondJump(cond, then_lbl, else_lbl, s.pos))

        self.block = self.func.add_block(then_lbl, s.pos)
        self.lower_stmt(s.then)
        if not self.block.is_terminated():
            self.block.add(IRJump(merge_lbl, s.pos))

        if s.els is not None:
            self.block = self.func.add_block(else_lbl, s.pos)
            self.lower_stmt(s.els)
            if not self.block.is_terminated():
                self.block.add(IRJump(merge_lbl, s.pos))

        self.block = self.func.add_block(merge_lbl, s.pos)
        
    def lower_for(self, s):
        self.push_scope()

        lo = self.lower_expr(s.lo)
        if lo is None:
            self.err("for loop lower bound has no value", s.pos)
        hi = self.lower_expr(s.hi)
        if hi is None:
            self.err("for loop upper bound has no value", s.pos)
        if not is_int(lo.type):
            self.err(f"for loop lower bound must be an integer, got "
                     f"'{ts(lo.type)}'", s.pos)
        if not is_int(hi.type):
            self.err(f"for loop upper bound must be an integer, got "
                     f"'{ts(hi.type)}'", s.pos)

        loop_cname = self.declare(s.var, lo.type)
        self.block.add(IRBind(loop_cname, lo, lo.type, True, s.pos))

        end_name = self.func.new_label("end")
        end_cname = self.declare(end_name, hi.type, False)
        self.block.add(IRBind(end_cname, hi, hi.type, False, s.pos))

        head_lbl = self.func.new_label("head")
        body_lbl = self.func.new_label("body")
        step_lbl = self.func.new_label("step")
        end_lbl  = self.func.new_label("end")

        self.block.add(IRJump(head_lbl, s.pos))

        self.block = self.func.add_block(head_lbl, s.pos)
        iv = IRLocal(loop_cname, lo.type)
        end_ref = IRLocal(end_cname, hi.type)
        cmp_op = '<=' if s.inclusive else '<'
        cond = self.func.new_temp(('prim', 'bool'))
        self.block.add(IRBinOp(cond, cmp_op, iv, end_ref,
                               ('prim', 'bool'), s.pos))
        self.block.add(IRCondJump(cond, body_lbl, end_lbl, s.pos))

        self.block = self.func.add_block(body_lbl, s.pos)
        self.loop_stack.append((step_lbl, end_lbl))
        self.lower_stmt(s.body)
        self.loop_stack.pop()
        if not self.block.is_terminated():
            self.block.add(IRJump(step_lbl, s.pos))

        self.block = self.func.add_block(step_lbl, s.pos)
        one = IRConst(1, lo.type)
        incremented = self.func.new_temp(lo.type)
        iv2 = IRLocal(loop_cname, lo.type)
        self.block.add(IRBinOp(incremented, '+', iv2, one, lo.type, s.pos))
        self.block.add(IRAssign(loop_cname, incremented, s.pos))
        self.block.add(IRJump(head_lbl, s.pos))

        self.block = self.func.add_block(end_lbl, s.pos)

        self.pop_scope()
        
        
    def lower_while(self, s):
        head_lbl = self.func.new_label("head")
        body_lbl = self.func.new_label("body")
        end_lbl  = self.func.new_label("end")

        self.block.add(IRJump(head_lbl, s.pos))

        self.block = self.func.add_block(head_lbl, s.pos)
        cond = self.lower_condition(s.cond, "while")
        self.block.add(IRCondJump(cond, body_lbl, end_lbl, s.pos))

        self.block = self.func.add_block(body_lbl, s.pos)
        self.loop_stack.append((head_lbl, end_lbl))
        self.lower_stmt(s.body)
        self.loop_stack.pop()
        if not self.block.is_terminated():
            self.block.add(IRJump(head_lbl, s.pos))

        self.block = self.func.add_block(end_lbl, s.pos)        
        
    
    def lower_expr(self, e):
        """Return an IRValue, or None for void expressions."""
        k = e.kind

        if k == 'int':
            return IRConst(e.value, ('prim', 'i32'))
        if k == 'float':
            return IRConst(e.value, ('prim', 'f64'))
        if k == 'float32':
            return IRConst(e.value, ('prim', 'f32'))
        if k == 'float16':
            return IRConst(e.value, ('prim', 'f16'))
        if k == 'bool':
            return IRConst(e.value, ('prim', 'bool'))
        if k == 'charlit':
            return IRConst(e.value, ('prim', 'char'))

        if k == 'strlit':
            if not e.parts:
                return IRConst("", ('prim', 'str'))
            acc = None
            for kind, v in e.parts:
                if kind == 'lit':
                    seg = IRConst(v, ('prim', 'str'))
                else:
                    inner = self.lower_expr(v)
                    if inner is None:
                        self.err("interpolated expression has no value", e.pos)
                    seg = self.coerce_to_str(inner, e.pos)
                if acc is None:
                    acc = seg
                else:
                    dst = self.func.new_temp(('prim', 'str'))
                    self.block.add(IRConcat(dst, acc, seg, e.pos))
                    acc = dst
            return acc
            
        if k == 'ident':
            r = self.lookup(e.name)
            if r is not None:
                cname, ty = r
                return IRLocal(cname, ty)
            if e.name in self.module.globals:
                _, ty, _val = self.module.globals[e.name]
                return IRLocal(e.name, ty)
            in_scope = []
            for scope in reversed(self.scopes):
                in_scope.extend(scope.keys())
            suggestion = ""
            if in_scope:
                suggestion = f" (in scope: {', '.join(sorted(in_scope))})"
            self.err(f"undefined name '{e.name}'{suggestion}", e.pos)

        if k == 'call':
            return self.lower_call(e)

        if k == 'bin':
            if e.op in ('&&', '||'):
                return self.lower_logical(e, e.op == '&&')
            l = self.lower_expr(e.left)
            r = self.lower_expr(e.right)
            if l is None or r is None:
                self.err(f"'{e.op}' needs value operands", e.pos)
            return self.lower_binop_values(l, r, e.op, e.pos)

        if k == 'un':
            if e.op == '&':
                return self.lower_ref(e)
            v = self.lower_expr(e.expr)
            if v is None:
                self.err(f"'{e.op}' needs a value operand", e.pos)
            if e.op == '!':
                if v.type != ('prim', 'bool'):
                    self.err(f"'!' requires a bool operand, got "
                             f"'{ts(v.type)}'", e.pos)
                dst = self.func.new_temp(('prim', 'bool'))
                self.block.add(IRUnOp(dst, '!', v, ('prim', 'bool'), e.pos))
                return dst
            dst = self.func.new_temp(v.type)
            self.block.add(IRUnOp(dst, e.op, v, v.type, e.pos))
            return dst

        if k == 'field':
            return self.lower_field(e)
            
        if k == 'veclit':
            return self.lower_veclit(e)
            
        if k == 'arrlit':
            return self.lower_arrlit(e)

        if k == 'alloc':
            return self.lower_alloc(e)

        if k == 'cast':
            return self.lower_cast(e)
            
        if k == 'index':
            return self.lower_index(e)

        self.not_impl(f"IR: expression '{k}' not yet lowered")

    # -- calls ---------------------------------------------------------------
    def lower_struct_init(self, name, e):
        declared = self.module.structs[name]
        declared_names = {fname for fname, _ in declared}

        given = {}
        for a in e.args:
            if a.kind != 'namedarg':
                self.err(f"struct '{name}' must be built with named fields, "
                         f"e.g. {name}(x: 1)", e.pos)
            if a.name in given:
                self.err(f"duplicate field '{a.name}'", e.pos)
            if a.name not in declared_names:
                self.err(f"struct '{name}' has no field '{a.name}'", e.pos)
            given[a.name] = a.value

        fields = []
        for fname, ftype in declared:
            if fname in given:
                v = self.lower_expr_as(
                    given[fname], ftype, given[fname].pos,
                    f"field '{fname}' of struct '{name}'")
                fields.append((fname, v))

        dst = self.func.new_temp(('named', name))
        self.block.add(IRStructInit(dst, fields, ('named', name), e.pos))
        return dst
    
    def lower_extern_call(self, name, e):
        params, ret = self.module.externs[name]
        if len(e.args) != len(params):
            self.err(f"'{name}' expects {len(params)} argument(s), "
                     f"got {len(e.args)}", e.pos)
        args = []
        for i, (a, (_, ptype)) in enumerate(zip(e.args, params), 1):
            if a.kind == 'namedarg':
                self.err(f"named arguments are not supported for extern "
                         f"'{name}'", a.pos)
            args.append(self.lower_expr_as(
                a, ptype, a.pos, f"argument {i} to '{name}'"))
        r = ret if ret else ('prim', 'void')
        dst = None
        if not (r[0] == 'prim' and r[1] == 'void'):
            dst = self.func.new_temp(r)
        # Externs keep their source name verbatim — no `lp_` prefix.
        self.block.add(IRCall(dst, name, args, r, e.pos))
        return dst
    
    def lower_field(self, e):
        # Enum value: Color.Red
        if e.base.kind == 'ident' and e.base.name in self.module.enums:
            variants = self.module.enums[e.base.name]
            if e.name not in variants:
                self.err(f"enum '{e.base.name}' has no variant '{e.name}'",
                         e.pos)
            cname = f"{e.base.name}_{e.name}"
            return IRConst(cname, ('named', e.base.name))

        base = self.lower_expr(e.base)
        if base is None:
            self.err("field access base has no value", e.pos)

        bt = base.type
        if bt[0] != 'named' or bt[1] not in self.module.structs:
            self.err(f"cannot access field '{e.name}' on "
                     f"'{bt[1] if bt[0] == 'named' else bt}'", e.pos)

        for fname, ftype in self.module.structs[bt[1]]:
            if fname == e.name:
                dst = self.func.new_temp(ftype)
                self.block.add(IRFieldGet(dst, base, fname, ftype, e.pos))
                return dst

        self.err(f"struct '{bt[1]}' has no field '{e.name}'", e.pos)
    
    def lower_cast(self, e):
        v = self.lower_expr(e.expr)
        if v is None:
            self.err("cannot cast a void expression", e.pos)
        src = v.type
        dst = self.resolve_type(e.type, e.pos)

        if src == dst:
            return v

        # Match the AST path: no casts to or from structs, enums, or
        # arrays. Those are the types where a raw C cast would be either
        # meaningless or a silent footgun.
        if src[0] in ('named', 'array'):
            self.err(f"cannot convert '{ts(src)}' to '{ts(dst)}'", e.pos)
        if dst[0] in ('named', 'array'):
            self.err(f"cannot convert '{ts(src)}' to '{ts(dst)}'", e.pos)

        tmp = self.func.new_temp(dst)
        self.block.add(IRCast(tmp, v, dst, e.pos))
        return tmp
     
    def lower_math(self, name, e):
        if len(e.args) != 1:
            self.err(f"{name} takes exactly one argument", e.pos)
        if e.args[0].kind == 'namedarg':
            self.err(f"{name} takes a positional argument", e.pos)

        v = self.lower_expr(e.args[0])
        if v is None:
            self.err(f"argument to {name} has no value", e.pos)
        at = v.type

        # Vector: same-type result, must be a float vector.
        if is_vector(at):
            if at not in (('prim', 'v4f32'),):
                self.err(f"{name} not supported on '{ts(at)}'", e.pos)
            dst = self.func.new_temp(at)
            self.block.add(IRMathOp(dst, name, [v], at, e.pos))
            return dst

        # Scalar: promote to f32 or f64, compute, return that.
        if at == ('prim', 'f32'):
            rt = ('prim', 'f32')
        elif at == ('prim', 'f64'):
            rt = ('prim', 'f64')
        elif is_num(at):
            # integer, f16, bf16, or q types promote to f64
            tmp = self.func.new_temp(('prim', 'f64'))
            self.block.add(IRCast(tmp, v, ('prim', 'f64'), e.pos))
            v = tmp
            rt = ('prim', 'f64')
        else:
            self.err(f"{name} needs a numeric argument, got '{ts(at)}'",
                     e.pos)

        dst = self.func.new_temp(rt)
        self.block.add(IRMathOp(dst, name, [v], rt, e.pos))
        return dst
        
    def lower_reduce(self, name, e):
        if len(e.args) != 1:
            self.err(f"{name} takes exactly one argument", e.pos)
        if e.args[0].kind == 'namedarg':
            self.err(f"{name} takes a positional argument", e.pos)

        v = self.lower_expr(e.args[0])
        if v is None:
            self.err(f"argument to {name} has no value", e.pos)

        if not is_vector(v.type):
            self.err(f"{name} needs a vector argument, got '{ts(v.type)}'",
                     e.pos)

        et = vector_elem(v.type)
        dst = self.func.new_temp(et)
        self.block.add(IRVecReduce(dst, VEC_REDUCTIONS[name], v, et, e.pos))
        return dst
        
    def lower_vec_binop(self, name, e):
        op, arity = VEC_BIN_OPS[name]
        if len(e.args) != arity:
            self.err(f"{name} takes exactly {arity} argument(s)", e.pos)

        args = []
        for a in e.args:
            if a.kind == 'namedarg':
                self.err(f"{name} takes positional arguments only", e.pos)
            v = self.lower_expr(a)
            if v is None:
                self.err(f"argument to {name} has no value", a.pos)
            if not is_vector(v.type):
                self.err(f"{name} needs vector arguments, "
                         f"got '{ts(v.type)}'", a.pos)
            args.append(v)

        vt = args[1].type if op == 'vselect' else args[0].type
        for a in args[1:]:
            if a.type != vt:
                self.err(f"{name} arguments must all have type "
                         f"'{ts(vt)}'", e.pos)
        if op == 'vselect' and args[0].type != vt:
            self.err(f"vselect mask must have type '{ts(vt)}'", e.pos)

        # vcmplt / vcmpgt produce the same vector shape, treating the
        # result as a bit-mask. Users pass it only to vselect.
        dst = self.func.new_temp(vt)
        self.block.add(IRVecBinOp(dst, op, args, vt, e.pos))
        return dst

    def lower_file_op(self, name, e):
        if name == 'file_size':
            if len(e.args) != 1:
                self.err("file_size takes 1 argument (path)", e.pos)
            path = self.lower_expr(e.args[0])
            if path is None or path.type != ('prim', 'str'):
                self.err("file_size: path must be a string", e.pos)
            dst = self.func.new_temp(('prim', 'i64'))
            self.block.add(IRCall(dst, 'lp_file_size', [path],
                                  ('prim', 'i64'), e.pos))
            return dst

        if name == 'read_file_at':
            if len(e.args) != 4:
                self.err("read_file_at takes 4 arguments "
                         "(path, offset, buf, n)", e.pos)
            path = self.lower_expr(e.args[0])
            if path is None or path.type != ('prim', 'str'):
                self.err("read_file_at: path must be a string", e.pos)
            offset = self.coerce_to_i64(e.args[1], "read_file_at: offset")
            buf = self.lower_expr(e.args[2])
            if buf is None or buf.type[0] != 'ptr':
                self.err("read_file_at: buf must be a pointer", e.pos)
            n = self.coerce_to_i64(e.args[3], "read_file_at: n")
            dst = self.func.new_temp(('prim', 'i64'))
            self.block.add(IRCall(dst, 'lp_read_file_at',
                                  [path, offset, buf, n],
                                  ('prim', 'i64'), e.pos))
            return dst

        if name == 'write_file':
            if len(e.args) != 3:
                self.err("write_file takes 3 arguments (path, buf, n)", e.pos)
            path = self.lower_expr(e.args[0])
            if path is None or path.type != ('prim', 'str'):
                self.err("write_file: path must be a string", e.pos)
            buf = self.lower_expr(e.args[1])
            if buf is None or buf.type[0] != 'ptr':
                self.err("write_file: buf must be a pointer", e.pos)
            n = self.coerce_to_i64(e.args[2], "write_file: n")
            dst = self.func.new_temp(('prim', 'i64'))
            self.block.add(IRCall(dst, 'lp_write_file', [path, buf, n],
                                  ('prim', 'i64'), e.pos))
            return dst

        self.err(f"unknown file builtin '{name}'", e.pos)

    def lower_read(self, name, e):
        if len(e.args) != 2:
            self.err(f"{name} takes 2 arguments (buf, offset)", e.pos)
        buf = self.lower_expr(e.args[0])
        if buf is None:
            self.err(f"{name}: buffer has no value", e.args[0].pos)
        if buf.type[0] != 'ptr':
            self.err(f"{name}: first argument must be a pointer, "
                     f"got '{ts(buf.type)}'", e.args[0].pos)
        off = self.coerce_to_i64(e.args[1], f"{name}: offset")
        ret_name, cname = READ_BUILTINS[name]
        dst = self.func.new_temp(('prim', ret_name))
        self.block.add(IRCall(dst, cname, [buf, off],
                              ('prim', ret_name), e.pos))
        return dst

    def lower_read_str(self, e):
        if len(e.args) != 3:
            self.err("read_str takes 3 arguments (buf, offset, n)", e.pos)
        buf = self.lower_expr(e.args[0])
        if buf is None:
            self.err("read_str: buffer has no value", e.args[0].pos)
        if buf.type[0] != 'ptr':
            self.err(f"read_str: first argument must be a pointer, "
                     f"got '{ts(buf.type)}'", e.args[0].pos)
        off = self.coerce_to_i64(e.args[1], "read_str: offset")
        n = self.coerce_to_i64(e.args[2], "read_str: n")
        dst = self.func.new_temp(('prim', 'str'))
        self.block.add(IRCall(dst, 'lp_read_str', [buf, off, n],
                              ('prim', 'str'), e.pos))
        return dst

    def lower_veclit(self, e):
        n = len(e.elems)
        vals = [self.lower_expr(el) for el in e.elems]
        for v, el in zip(vals, e.elems):
            if v is None:
                self.err("vector element has no value", el.pos)

        # Common element type across all slots. Integers and floats of
        # the same width promote; mixed integer/float promotes to float.
        common = vals[0].type
        for v in vals[1:]:
            common = self.common_type(common, v.type, e.pos)
        if common[0] != 'prim':
            self.err(f"vector element type '{ts(common)}' not supported",
                     e.pos)

        vt = vector_of(common[1], n)
        if vt is None:
            self.err(f"no vector type for {n} × '{ts(common)}'", e.pos)

        elems = []
        for v in vals:
            if v.type == common:
                elems.append(v)
            else:
                tmp = self.func.new_temp(common)
                self.block.add(IRCast(tmp, v, common, e.pos))
                elems.append(tmp)

        dst = self.func.new_temp(vt)
        self.block.add(IRVecLit(dst, elems, vt, e.pos))
        return dst
        
    def lower_arrlit(self, e):
        vals = [self.lower_expr(el) for el in e.elems]
        for v, el in zip(vals, e.elems):
            if v is None:
                self.err("array element has no value", el.pos)

        common = vals[0].type
        for v in vals[1:]:
            common = self.common_type(common, v.type, e.pos)

        elems = []
        for v in vals:
            if v.type == common:
                elems.append(v)
            else:
                tmp = self.func.new_temp(common)
                self.block.add(IRCast(tmp, v, common, e.pos))
                elems.append(tmp)

        at = ('array', common, len(elems))
        dst = self.func.new_temp(at)
        self.block.add(IRArrayLit(dst, elems, at, e.pos))
        return dst

    def lower_arrlit_as(self, e, target, pos):
        """Lower an array literal directly against a declared array type."""
        elem = array_elem(target)
        count = array_len(target)
        if len(e.elems) != count:
            self.err(f"array literal has {len(e.elems)} element(s), "
                     f"expected {count}", e.pos)
        elems = [self.lower_expr_as(el, elem, el.pos, "array element")
                 for el in e.elems]
        dst = self.func.new_temp(target)
        self.block.add(IRArrayLit(dst, elems, target, pos or e.pos))
        return dst

    def lower_veclit_as(self, e, target, pos):
        """Lower a vector literal directly against a declared vector type."""
        elem = vector_elem(target)
        count = vector_len(target)
        if len(e.elems) != count:
            self.err(f"vector literal has {len(e.elems)} element(s), "
                     f"expected {count}", e.pos)
        elems = [self.lower_expr_as(el, elem, el.pos, "vector element")
                 for el in e.elems]
        dst = self.func.new_temp(target)
        self.block.add(IRVecLit(dst, elems, target, pos or e.pos))
        return dst
        
    def broadcast(self, v, vec_type, pos):
        """Cast scalar v to vec_type's element type and splat it."""
        et = vector_elem(vec_type)
        if v.type != et:
            tmp = self.func.new_temp(et)
            self.block.add(IRCast(tmp, v, et, pos))
            v = tmp
        dst = self.func.new_temp(vec_type)
        self.block.add(IRSplat(dst, v, vec_type, pos))
        return dst

    def lower_index(self, e):
        base = self.lower_expr(e.base)
        if base is None:
            self.err("index base has no value", e.pos)
        idx = self.lower_expr(e.index)
        if idx is None:
            self.err("index expression has no value", e.pos)
        if not is_int(idx.type):
            self.err(f"index must be an integer, got '{ts(idx.type)}'",
                     e.pos)

        bt = base.type
        if is_vector(bt):
            et = vector_elem(bt)
            dst = self.func.new_temp(et)
            self.block.add(IRVecIndex(dst, base, idx, et, e.pos))
            return dst

        if is_array(bt):
            et = array_elem(bt)
            dst = self.func.new_temp(et)
            self.block.add(IRArrayIndex(dst, base, idx, et, e.pos))
            return dst

        # ptr[[T; N]] — index through the pointer without an explicit
        # dereference. The emitter renders base->e[i] instead of base.e[i].
        if bt[0] == 'ptr' and is_array(bt[1]):
            et = array_elem(bt[1])
            dst = self.func.new_temp(et)
            self.block.add(IRArrayIndex(dst, base, idx, et, e.pos))
            return dst

        if bt[0] == 'ptr' and bt[1][0] in ('prim', 'ptr', 'named'):
            et = bt[1]
            dst = self.func.new_temp(et)
            self.block.add(IRArrayIndex(dst, base, idx, et, e.pos))
            return dst

        if bt == ('prim', 'str'):
            dst = self.func.new_temp(('prim', 'char'))
            self.block.add(IRArrayIndex(dst, base, idx, ('prim', 'char'), e.pos))
            return dst

        self.not_impl(f"IR: indexing '{ts(bt)}' not yet lowered")

    def lower_ptr_arith_values(self, l, r, op, pos):
        lt = l.type
        rt = r.type
        lp = lt[0] == 'ptr'
        rp = rt[0] == 'ptr'

        # ptr - ptr → i64 (element difference)
        if lp and rp and op == '-':
            if lt != rt:
                self.err(f"cannot subtract '{ts(rt)}' from '{ts(lt)}'", pos)
            dst = self.func.new_temp(('prim', 'i64'))
            self.block.add(IRBinOp(dst, '-', l, r, ('prim', 'i64'), pos))
            return dst

        # ptr ± int → same ptr type
        if lp and not rp:
            if not is_int(rt):
                self.err(f"pointer arithmetic needs an integer, "
                         f"got '{ts(rt)}'", pos)
            dst = self.func.new_temp(lt)
            self.block.add(IRBinOp(dst, op, l, r, lt, pos))
            return dst

        # int + ptr → same ptr type (commutative; - is not)
        if rp and not lp:
            if op != '+':
                self.err(f"cannot use '{op}' with pointer on the right", pos)
            if not is_int(lt):
                self.err(f"pointer arithmetic needs an integer, "
                         f"got '{ts(lt)}'", pos)
            dst = self.func.new_temp(rt)
            self.block.add(IRBinOp(dst, '+', l, r, rt, pos))
            return dst

        self.err(f"unsupported pointer arithmetic between "
                 f"'{ts(lt)}' and '{ts(rt)}'", pos)

    def lower_ref(self, e):
        if e.expr.kind != 'ident':
            self.err("'&' requires a named value", e.pos)
        name = e.expr.name
        r = self.lookup(name)
        if r is None:
            self.err(f"undefined name '{name}'", e.pos)
        cname, t = r
        dst = self.func.new_temp(('ptr', t))
        self.block.add(IRRef(dst, IRLocal(cname, t), ('ptr', t), e.pos))
        return dst

    def lower_alloc(self, e):
        t = self.resolve_type(e.type, e.pos)
        if not is_array(t):
            self.err(f"alloc requires an array type, got '{ts(t)}'", e.pos)
        dst = self.func.new_temp(('ptr', t))
        self.block.add(IRAlloc(dst, t, e.pos))
        return dst

    def lower_call(self, e):
        if e.callee.kind != 'ident':
            self.not_impl("IR: only named calls are supported")

        name = e.callee.name

        if name in BUILTIN_C_NAMES:
            return self.lower_builtin(name, e)

        if name in MATH_INTRINSICS:
            return self.lower_math(name, e) 
            
        if name in VEC_REDUCTIONS:
            return self.lower_reduce(name, e)
            
        if name in VEC_BIN_OPS:
            return self.lower_vec_binop(name, e)

        if name in FILE_BUILTINS:
            return self.lower_file_op(name, e)

        if name in READ_BUILTINS:
            return self.lower_read(name, e)

        if name in STRING_BUILTINS:
            return self.lower_read_str(e)
            
        if name in self.module.structs:
            return self.lower_struct_init(name, e)

        if name in self.module.enums:
            self.not_impl(f"IR: enum value '{name}' not yet lowered")

        f = self.func_by_name(name)
        if f is None:
            if name in self.module.externs:
                return self.lower_extern_call(name, e)
            self.err(f"call to undefined function '{name}'", e.pos)

        if len(e.args) != len(f.params):
            self.err(f"'{name}' expects {len(f.params)} argument(s), "
                     f"got {len(e.args)}", e.pos)

        args = []
        for i, (a, (_, ptype)) in enumerate(zip(e.args, f.params), 1):
            if a.kind == 'namedarg':
                self.err(f"named arguments are not supported for function "
                         f"'{name}'", a.pos)
            args.append(self.lower_expr_as(
                a, ptype, a.pos, f"argument {i} to '{name}'"))

        ret = f.ret if f.ret else ('prim', 'void')
        dst = None
        if ret != ('prim', 'void'):
            dst = self.func.new_temp(ret)
        self.block.add(IRCall(dst, f.cname, args, ret, e.pos))
        return dst

    def lower_builtin(self, name, e):
        cname = BUILTIN_C_NAMES[name]

        if name in ('print', 'println'):
            if len(e.args) == 0:
                arg = IRConst("", ('prim', 'str'))
            elif len(e.args) == 1:
                v = self.lower_expr(e.args[0])
                if v is None:
                    self.err(f"argument to '{name}' has no value", e.pos)
                arg = self.coerce_to_str(v, e.args[0].pos)
            else:
                self.err(f"{name} takes at most one argument", e.pos)
            self.block.add(IRCall(None, cname, [arg], ('prim', 'void'), e.pos))
            return None

        if name == 'panic':
            if len(e.args) != 1:
                self.err("panic takes exactly one argument", e.pos)
            v = self.lower_expr(e.args[0])
            if v is None:
                self.err("argument to panic has no value", e.pos)
            arg = self.coerce_to_str(v, e.args[0].pos)
            self.block.add(IRCall(None, cname, [arg], ('prim', 'void'), e.pos))
            return None

        self.not_impl(f"IR: builtin '{name}' not yet lowered")

        
    def coerce_to_str(self, v, pos):
        """Return a str-typed IRValue equivalent to v. Emits an IRToString
        for anything that is not already a str."""
        t = v.type
        if t == ('prim', 'str'):
            return v
        if is_vector(t):
            self.err(f"cannot stringify '{ts(t)}'; extract elements first",
                     pos)
        if t == ('prim', 'bool') or t == ('prim', 'char') \
                or t in (('prim', 'f16'), ('prim', 'bf16'),
                         ('prim', 'f32'), ('prim', 'f64'),
                         ('prim', 'q7'), ('prim', 'q15'), ('prim', 'q31')) \
                or is_int(t) or t[0] == 'ptr':
            dst = self.func.new_temp(('prim', 'str'))
            self.block.add(IRToString(dst, v, t, pos))
            return dst
        self.err(f"cannot convert '{t}' to str", pos)

    def coerce_to_i64(self, node, ctx):
        v = self.lower_expr(node)
        if v is None:
            self.err(f"{ctx} has no value", node.pos)
        if v.type == ('prim', 'i64'):
            return v
        if not is_int(v.type):
            self.err(f"{ctx} must be an integer, got '{ts(v.type)}'",
                     node.pos)
        tmp = self.func.new_temp(('prim', 'i64'))
        self.block.add(IRCast(tmp, v, ('prim', 'i64'), node.pos))
        return tmp
    # -- types ---------------------------------------------------------------
        
    def common_type(self, a, b, pos):
        if a == b:
            return a
        # Fixed-point does not auto-promote. `q15 + q15` stays q15
        # (handled above by the identity check), but anything else
        # involving a q type widens to f32. Users who want q-to-q
        # arithmetic should use an explicit cast.
        from lepton_types import is_quant
        if is_quant(a) or is_quant(b):
            return ('prim', 'f32')
        if is_vector(a) or is_vector(b):
            self.err(f"unsupported vector operation between "
                     f"'{ts(a)}' and '{ts(b)}'", pos)
        if is_num(a) and is_num(b):
            return ('prim', a[1] if RANK[a[1]] >= RANK[b[1]] else b[1])
        self.err(f"incompatible types '{a}' and '{b}'", pos)        


def is_int(t):
    from lepton_types import is_int as _is_int
    return _is_int(t)


def lower_module(items, fname, strict_types=False):
    """Entry point: AST item list -> IRModule."""
    lowering = Lowering(items, fname, strict_types)
    module = lowering.lower()
    module.warnings = lowering.warnings
    return module
