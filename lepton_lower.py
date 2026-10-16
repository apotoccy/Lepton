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
    IRModule, IRFunc, IRBlock, IRTemp, IRLocal, IRConst,
    IRBind, IRAssign, IRBinOp, IRUnOp, IRCall,
    IRConcat, IRToString, IRCast, IRStructInit, IRFieldGet,
    IRVecLit, IRVecIndex,
    IRReturn, IRJump, IRCondJump, IRNotImplemented,
)

from leptonc import CompileError
from lepton_types import (is_num, RANK, PRIMS, ts,
                          is_vector, vector_elem, vector_len, vector_of)


# Source-level builtin name -> C-level name.
BUILTIN_C_NAMES = {
    'print':   'lp_print',
    'println': 'lp_println',
    'panic':   'lp_panic',
}


class Lowering:
    def __init__(self, items, fname):
        self.items = items
        self.fname = fname
        self.module = IRModule()
        self.func = None
        self.block = None
        self.scopes = []
        self.loop_stack = []

    # -- diagnostics ---------------------------------------------------------
    def err(self, msg, pos=None):
        if pos is None:
            raise CompileError(msg, 0, 0, self.fname)
        raise CompileError(msg, pos[0], pos[1], self.fname)

    def not_impl(self, msg):
        raise IRNotImplemented(msg)

    # -- driver --------------------------------------------------------------
    def lower(self):
        # Pass 1: collect signatures and nominal types.
        for it in self.items:
            if it.kind == 'fn':
                cname = 'main' if it.name == 'main' else 'lp_' + it.name
                f = IRFunc(it.name,
                           [(p.name, p.type) for p in it.params],
                           it.ret, it.pos)
                f.cname = cname
                self.module.funcs.append(f)
            elif it.kind == 'extern':
                self.module.externs[it.name] = (
                    [(p.name, p.type) for p in it.params], it.ret)
            elif it.kind == 'struct':
                self.module.structs[it.name] = list(it.fields)
            elif it.kind == 'enum':
                self.module.enums[it.name] = list(it.variants)
            elif it.kind == 'const':
                self.module.globals[it.name] = (it.name, it.type)
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

    def pop_scope(self):
        self.scopes.pop()

    def declare(self, name, type):
        self.scopes[-1][name] = type

    def lookup(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
    # -- functions -----------------------------------------------------------
    def lower_fn(self, it, f):
        self.func = f
        self.scopes = [{p[0]: p[1] for p in f.params}]
        self.loop_stack = []
        self.block = f.add_block('entry', it.pos)
        for s in it.body.stmts:
            self.lower_stmt(s)
        if not self.block.is_terminated():
            if it.name == 'main':
                self.block.add(IRReturn(IRConst(0, ('prim', 'i32')), it.pos))
            else:
                self.block.add(IRReturn(None, it.pos))
                
                
                

    # -- statements ----------------------------------------------------------
    def lower_stmt(self, s):
        if self.block.is_terminated():
            return
        k = s.kind

        if k == 'exprstmt':
            self.lower_expr(s.expr)
            return

        if k == 'return':
            if s.value is None:
                self.block.add(IRReturn(None, s.pos))
            else:
                v = self.lower_expr(s.value)
                if v is None:
                    self.err("cannot return a void expression", s.pos)
                self.block.add(IRReturn(v, s.pos))
            return
            
        if k == 'block':
            self.push_scope()
            for sub in s.stmts:
                self.lower_stmt(sub)
            self.pop_scope()
            return

        if k == 'let':
            if s.value is None:
                self.not_impl("IR: `let` without initialiser")
            if s.type is not None:
                self.resolve_type(s.type, s.pos)
            v = self.lower_expr(s.value)
            if v is None:
                self.err("cannot bind a void expression to a name", s.pos)
            self.block.add(IRBind(s.name, v, v.type, s.mutable, s.pos))
            self.declare(s.name, v.type)
            return

        if k == 'assign':
            if s.op != '=':
                self.not_impl(f"IR: compound assignment '{s.op}'")
            if s.target.kind != 'ident':
                self.not_impl("IR: non-ident assignment target")
            v = self.lower_expr(s.value)
            if v is None:
                self.err("cannot assign a void expression", s.pos)
            self.block.add(IRAssign(s.target.name, v, s.pos))
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

    # -- expressions ---------------------------------------------------------
    def lower_logical(self, e, is_and):
        l = self.lower_expr(e.left)
        if l is None:
            self.err(f"'{e.op}' needs value operands", e.pos)

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
        self.block.add(IRAssign(dst_name, r, e.pos))
        self.block.add(IRJump(merge_lbl, e.pos))

        # Short-circuit path: dst retains its placeholder value.
        self.block = self.func.add_block(short_lbl, e.pos)
        self.block.add(IRJump(merge_lbl, e.pos))

        self.block = self.func.add_block(merge_lbl, e.pos)
        return IRLocal(dst_name, ('prim', 'bool'))
    
    def lower_if(self, s):
        cond = self.lower_expr(s.cond)
        if cond is None:
            self.err("if condition has no value", s.pos)

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

        # The loop variable. Mutable so the increment can update it.
        self.block.add(IRBind(s.var, lo, lo.type, True, s.pos))
        self.declare(s.var, lo.type)

        # The upper bound is evaluated once, before the loop. This matches
        # the AST emitter and is what users expect from a range for-loop.
        end_name = self.func.new_label("end")     # unique C-safe name
        self.block.add(IRBind(end_name, hi, hi.type, False, s.pos))
        self.declare(end_name, hi.type)

        head_lbl = self.func.new_label("head")
        body_lbl = self.func.new_label("body")
        step_lbl = self.func.new_label("step")
        end_lbl  = self.func.new_label("end")

        self.block.add(IRJump(head_lbl, s.pos))

        # Condition: i < end (or i <= end).
        self.block = self.func.add_block(head_lbl, s.pos)
        iv = IRLocal(s.var, lo.type)
        end_ref = IRLocal(end_name, hi.type)
        cmp_op = '<=' if s.inclusive else '<'
        cond = self.func.new_temp(('prim', 'bool'))
        self.block.add(IRBinOp(cond, cmp_op, iv, end_ref,
                               ('prim', 'bool'), s.pos))
        self.block.add(IRCondJump(cond, body_lbl, end_lbl, s.pos))

        # Body. `continue` jumps to the step block, not the head, so the
        # increment still runs.
        self.block = self.func.add_block(body_lbl, s.pos)
        self.loop_stack.append((step_lbl, end_lbl))
        self.lower_stmt(s.body)
        self.loop_stack.pop()
        if not self.block.is_terminated():
            self.block.add(IRJump(step_lbl, s.pos))

        # Step: i = i + 1, then back to head.
        self.block = self.func.add_block(step_lbl, s.pos)
        one = IRConst(1, lo.type)
        incremented = self.func.new_temp(lo.type)
        iv2 = IRLocal(s.var, lo.type)
        self.block.add(IRBinOp(incremented, '+', iv2, one, lo.type, s.pos))
        self.block.add(IRAssign(s.var, incremented, s.pos))
        self.block.add(IRJump(head_lbl, s.pos))

        # Merge.
        self.block = self.func.add_block(end_lbl, s.pos)

        self.pop_scope()
        
        
    def lower_while(self, s):
        head_lbl = self.func.new_label("head")
        body_lbl = self.func.new_label("body")
        end_lbl  = self.func.new_label("end")

        self.block.add(IRJump(head_lbl, s.pos))

        self.block = self.func.add_block(head_lbl, s.pos)
        cond = self.lower_expr(s.cond)
        if cond is None:
            self.err("while condition has no value", s.pos)
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
            t = self.lookup(e.name)
            if t is not None:
                return IRLocal(e.name, t)
            if e.name in self.module.globals:
                _, ty = self.module.globals[e.name]
                return IRLocal(e.name, ty)
            self.err(f"undefined name '{e.name}'", e.pos)

        if k == 'call':
            return self.lower_call(e)

            
        if k == 'bin':
            if e.op in ('&&', '||'):
                return self.lower_logical(e, e.op == '&&')
            l = self.lower_expr(e.left)
            r = self.lower_expr(e.right)
            if l is None or r is None:
                self.err(f"'{e.op}' needs value operands", e.pos)
            # Comparison operators produce bool, not the common type of
            # their operands. Ordering comparisons additionally require
            # numeric or pointer operands, matching the AST path.
            if e.op in ('==', '!='):
                self.common_type(l.type, r.type, e.pos)
                t = ('prim', 'bool')
            elif e.op in ('<', '<=', '>', '>='):
                ct = self.common_type(l.type, r.type, e.pos)
                if not is_num(ct) and ct[0] != 'ptr':
                    self.err(f"cannot order '{ts(l.type)}' and "
                             f"'{ts(r.type)}'", e.pos)
                t = ('prim', 'bool')
            else:
                t = self.common_type(l.type, r.type, e.pos)
            dst = self.func.new_temp(t)
            self.block.add(IRBinOp(dst, e.op, l, r, t, e.pos))
            return dst

        if k == 'un':
            v = self.lower_expr(e.expr)
            if v is None:
                self.err(f"'{e.op}' needs a value operand", e.pos)
            dst = self.func.new_temp(v.type)
            self.block.add(IRUnOp(dst, e.op, v, v.type, e.pos))
            return dst

        if k == 'field':
            return self.lower_field(e)
            
        if k == 'veclit':
            return self.lower_veclit(e)

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
        for fname, _ in declared:
            if fname in given:
                v = self.lower_expr(given[fname])
                if v is None:
                    self.err(f"field '{fname}' has no value",
                             given[fname].pos)
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
        for a, (_, ptype) in zip(e.args, params):
            v = self.lower_expr(a)
            if v is None:
                self.err(f"argument to '{name}' has no value", a.pos)
            args.append(v)
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

        self.not_impl(f"IR: indexing '{ts(bt)}' not yet lowered")
    
    def lower_call(self, e):
        if e.callee.kind != 'ident':
            self.not_impl("IR: only named calls are supported")

        name = e.callee.name

        if name in BUILTIN_C_NAMES:
            return self.lower_builtin(name, e)  
            
        if name in self.module.structs:
            return self.lower_struct_init(name, e)

        if name in self.module.enums:
            self.not_impl(f"IR: enum value '{name}' not yet lowered")

        f = self.func_by_name(name)
        if f is None:
            if name in self.module.externs:
                return self.lower_extern_call(name, e)
            self.err(f"call to undefined function '{name}'", e.pos)

        args = []
        for a in e.args:
            v = self.lower_expr(a)
            if v is None:
                self.err(f"argument to '{name}' has no value", a.pos)
            args.append(v)

        ret = f.ret if f.ret else ('prim', 'void')
        dst = None
        if ret[1] != 'void':
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


def lower_module(items, fname):
    """Entry point: AST item list -> IRModule."""
    return Lowering(items, fname).lower()
