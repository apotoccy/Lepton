"""
IR verification.

Runs after lowering and before C emission. The verifier checks the
invariants the emitter assumes, so a lowering bug is reported as a
Lepton error instead of surfacing later as invalid C or a crash.

Checks include:

  - unique block labels, valid jump targets, and terminator placement
  - unique locals, parameter types, and function return types
  - every instruction's destination and operand types
  - call arity and argument/return types for user functions, externs,
    and the runtime helpers the lowerer emits directly
  - array/vector literal shapes, indexing, struct fields, and references
  - no reachable fallthrough from a non-void function

The verifier deliberately does not re-implement source-level conversion
policy; it checks the typed IR that lowering produced.
"""
from __future__ import annotations

from lepton_ir import (
    IRTemp, IRLocal, IRConst,
    IRBind, IRAssign, IRBinOp, IRUnOp, IRCall,
    IRConcat, IRToString, IRCast, IRMathOp, IRVecReduce, IRSplat,
    IRVecBinOp, IRArrayLit, IRArrayIndex, IRArraySet, IRRef, IRAlloc,
    IRStructInit, IRFieldGet,
    IRVecLit, IRVecIndex, IRVecSet,
    IRReturn, IRJump, IRCondJump, TERMINATORS,
)
from lepton_types import (
    is_int, is_quant, is_num, is_array, array_elem, array_len,
    is_vector, vector_elem, vector_len, ts,
)
from leptonc import CompileError


VOID = ('prim', 'void')
STR = ('prim', 'str')
BOOL = ('prim', 'bool')
I64 = ('prim', 'i64')
I32 = ('prim', 'i32')


class IRStats:
    """Counts collected while verifying a module."""

    def __init__(self, functions, blocks, instructions):
        self.functions = functions
        self.blocks = blocks
        self.instructions = instructions

    def __str__(self):
        return (f"{self.functions} function(s), {self.blocks} block(s), "
                f"{self.instructions} instruction(s)")


class _AnyPtr:
    """Sentinel for runtime helpers that accept any pointer."""


ANY_PTR = _AnyPtr()


def _is_void(t):
    return t is None or t == VOID


def _is_stringifiable(t):
    if t is None or is_vector(t) or is_array(t):
        return False
    if t[0] == 'named':
        return False
    if t[0] == 'ptr':
        return True
    if t in (STR, BOOL, ('prim', 'char')):
        return True
    return is_num(t)


def _is_pointer(t):
    return t is not None and t[0] == 'ptr'


def _is_struct(module, t):
    return (t is not None and t[0] == 'named'
            and t[1] in module.structs)


def _field_type(module, struct_name, field):
    for fname, ftype in module.structs[struct_name]:
        if fname == field:
            return ftype
    return None


def _array_index_elem(t):
    """Element type of an IRArrayIndex/IRArraySet base."""
    if t is None:
        return None
    if t == ('prim', 'str'):
        return ('prim', 'char')
    if is_array(t):
        return array_elem(t)
    if t[0] == 'ptr':
        inner = t[1]
        if is_array(inner):
            return array_elem(inner)
        if inner[0] in ('prim', 'ptr', 'named'):
            return inner
    return None


# IRCall callees emitted by the lowerer for builtins (not user functions
# or externs). `expected` uses ANY_PTR for byte-buffer parameters.
RUNTIME_CALLS = {
    'lp_print':    ([STR], VOID),
    'lp_println':  ([STR], VOID),
    'lp_panic':    ([STR], VOID),
    'lp_file_size': ([STR], I64),
    'lp_read_file_at': ([STR, I64, ANY_PTR, I64], I64),
    'lp_write_file': ([STR, ANY_PTR, I64], I64),
    'lp_read_str': ([ANY_PTR, I64, I64], STR),
    'lp_rd_u8':    ([ANY_PTR, I64], ('prim', 'u8')),
    'lp_rd_u16_le': ([ANY_PTR, I64], ('prim', 'u16')),
    'lp_rd_u32_le': ([ANY_PTR, I64], ('prim', 'u32')),
    'lp_rd_u64_le': ([ANY_PTR, I64], ('prim', 'u64')),
    'lp_rd_i8_le': ([ANY_PTR, I64], ('prim', 'i8')),
    'lp_rd_i32_le': ([ANY_PTR, I64], ('prim', 'i32')),
    'lp_rd_i64_le': ([ANY_PTR, I64], ('prim', 'i64')),
    'lp_rd_f16_le': ([ANY_PTR, I64], ('prim', 'f32')),
    'lp_rd_f32_le': ([ANY_PTR, I64], ('prim', 'f32')),
}


class Verifier:
    def __init__(self, module, fname="<ir>"):
        self.module = module
        self.fname = fname
        self.local_types = {}
        self.temp_types = {}
        self.global_types = {}
        self.labels = {}
        self.func = None
        self.block = None
        self.stats = IRStats(0, 0, 0)

    # -- diagnostics --------------------------------------------------------
    def err(self, msg, pos=None):
        line, col = pos if pos else (0, 0)
        where = f"function '{self.func.name}'" if self.func else "module"
        raise CompileError(f"IR verification failed in {where}: {msg}",
                           line, col, self.fname)

    # -- module -------------------------------------------------------------
    def verify(self):
        cnames = set()
        for name, entry in self.module.globals.items():
            if len(entry) == 3:
                cname, ty, _value = entry
            else:
                cname, ty = entry
            self.global_types[cname] = ty

        for f in self.module.funcs:
            if f.cname is None:
                self.err("function has no C name", f.pos)
            if f.cname in cnames:
                self.err(f"duplicate C symbol '{f.cname}'", f.pos)
            cnames.add(f.cname)
            self.verify_func(f)

        for name, (params, ret) in self.module.externs.items():
            for pname, ptype in params:
                if ptype is None or ptype == VOID:
                    self.err(f"extern '{name}' parameter '{pname}' has "
                             f"invalid type", None)
        return self.stats

    def verify_func(self, f):
        self.func = f
        self.local_types = {}
        self.temp_types = {}
        self.labels = {}

        if not f.blocks:
            self.err("function has no blocks", f.pos)
        if f.entry is None or f.entry is not f.blocks[0]:
            self.err("entry must be the first block", f.pos)

        for pname, ptype in f.params:
            if ptype is None or ptype == VOID:
                self.err(f"parameter '{pname}' has invalid type", f.pos)
            if pname in self.local_types:
                self.err(f"duplicate parameter '{pname}'", f.pos)
            self.local_types[pname] = ptype

        if f.ret is not None and f.ret == VOID:
            f_ret = VOID
        else:
            f_ret = f.ret
        if f.name == 'main' and _is_void(f_ret):
            # The lowerer gives main an implicit i32 return.
            f_ret = I32

        for b in f.blocks:
            self.stats.blocks += 1
            if b.label in self.labels:
                self.err(f"duplicate block label '{b.label}'", b.pos)
            self.labels[b.label] = b

        for b in f.blocks:
            for instr in b.instrs:
                self.stats.instructions += 1
                self.collect_defs(instr)

        for b in f.blocks:
            self.block = b
            self.verify_block(b)

        self.verify_fallthrough(f, f_ret)

        self.stats.functions += 1
        self.local_types = {}
        self.temp_types = {}
        self.labels = {}

    # -- definitions --------------------------------------------------------
    def collect_defs(self, instr):
        if isinstance(instr, IRBind):
            if not isinstance(instr.name, str) or not instr.name:
                self.err("IRBind has an invalid name", instr.pos)
            if instr.type is None or instr.type == VOID:
                self.err(f"local '{instr.name}' has invalid type", instr.pos)
            if instr.name in self.local_types:
                self.err(f"duplicate local '{instr.name}'", instr.pos)
            self.local_types[instr.name] = instr.type
            return

        if not hasattr(instr, 'dst'):
            return
        dst = instr.dst
        if dst is None:
            if not isinstance(instr, IRCall):
                self.err(f"{type(instr).__name__} has no destination",
                         instr.pos)
            return
        if not isinstance(dst, IRTemp):
            self.err(f"{type(instr).__name__} destination is not an IRTemp",
                     instr.pos)
        if dst.type is None or dst.type == VOID:
            self.err(f"temporary %{dst.id} has invalid type", instr.pos)
        if dst.id in self.temp_types:
            self.err(f"duplicate temporary %{dst.id}", instr.pos)
        self.temp_types[dst.id] = dst.type

    # -- blocks -------------------------------------------------------------
    def verify_block(self, b):
        term_at = None
        for i, instr in enumerate(b.instrs):
            if isinstance(instr, TERMINATORS):
                term_at = i
                break
        if term_at is not None and term_at != len(b.instrs) - 1:
            self.err(f"block '{b.label}' has instructions after a terminator",
                     b.instrs[term_at + 1].pos)
        if term_at is None and b is not self.func.blocks[-1]:
            self.err(f"block '{b.label}' has no terminator", b.pos)

        for instr in b.instrs:
            self.verify_instr(instr)

    def verify_fallthrough(self, f, f_ret):
        last = f.blocks[-1]
        if last.is_terminated():
            return
        if last.label not in self.reachable_labels(f):
            return
        if _is_void(f_ret):
            return
        self.err(f"reachable block '{last.label}' falls through without a "
                 f"return", last.pos)

    def reachable_labels(self, f):
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
                    stack.append(self.labels.get(instr.target))
                    break
                if isinstance(instr, IRCondJump):
                    stack.append(self.labels.get(instr.then_target))
                    stack.append(self.labels.get(instr.else_target))
                    break
        return seen

    # -- values -------------------------------------------------------------
    def check_value(self, v, what="operand"):
        if v is None:
            self.err(f"{what} is missing")
        if v.type is None or v.type == VOID:
            self.err(f"{what} has invalid type")
        if isinstance(v, IRTemp):
            actual = self.temp_types.get(v.id)
            if actual is None:
                self.err(f"{what} uses undefined temporary %{v.id}")
            if actual != v.type:
                self.err(f"{what} temporary %{v.id} has type "
                         f"'{ts(v.type)}', defined as '{ts(actual)}'")
        elif isinstance(v, IRLocal):
            actual = self.local_types.get(v.name)
            if actual is None:
                actual = self.global_types.get(v.name)
            if actual is None:
                self.err(f"{what} uses undefined local '{v.name}'")
            if actual != v.type:
                self.err(f"{what} local '{v.name}' has type "
                         f"'{ts(v.type)}', defined as '{ts(actual)}'")
        elif isinstance(v, IRConst):
            pass
        else:
            self.err(f"{what} is not an IR value: {v!r}")

    def check_assignable(self, value, target, what):
        self.check_value(value, what)
        if value.type != target:
            self.err(f"{what} has type '{ts(value.type)}', "
                     f"expected '{ts(target)}'")

    def check_index(self, idx):
        self.check_value(idx, "index")
        if not is_int(idx.type):
            self.err(f"index has type '{ts(idx.type)}', expected an integer")

    def check_dst(self, instr, expected):
        dst = instr.dst
        if not isinstance(dst, IRTemp):
            self.err(f"{type(instr).__name__} destination is not a temporary",
                     instr.pos)
        if dst.type != expected:
            self.err(f"{type(instr).__name__} destination has type "
                     f"'{ts(dst.type)}', expected '{ts(expected)}'",
                     instr.pos)

    # -- instructions -------------------------------------------------------
    def verify_instr(self, instr):
        if isinstance(instr, IRBind):
            self.check_assignable(instr.value, instr.type,
                                  f"initializer for '{instr.name}'")
            return

        if isinstance(instr, IRAssign):
            target = self.local_types.get(instr.name)
            if target is None:
                self.err(f"assignment to undefined local '{instr.name}'",
                         instr.pos)
            self.check_assignable(instr.value, target,
                                  f"assignment to '{instr.name}'")
            return

        if isinstance(instr, IRBinOp):
            self.verify_binop(instr)
            return

        if isinstance(instr, IRUnOp):
            self.verify_unop(instr)
            return

        if isinstance(instr, IRCall):
            self.verify_call(instr)
            return

        if isinstance(instr, IRConcat):
            self.check_dst(instr, STR)
            self.check_assignable(instr.left, STR, "left concat operand")
            self.check_assignable(instr.right, STR, "right concat operand")
            return

        if isinstance(instr, IRToString):
            self.check_dst(instr, STR)
            self.check_value(instr.value, "stringified value")
            if instr.src_type != instr.value.type:
                self.err(f"toString source type '{ts(instr.src_type)}' does "
                         f"not match value type '{ts(instr.value.type)}'",
                         instr.pos)
            if not _is_stringifiable(instr.value.type):
                self.err(f"cannot stringify '{ts(instr.value.type)}'",
                         instr.pos)
            return

        if isinstance(instr, IRCast):
            if instr.type is None or instr.type == VOID:
                self.err("cast target has invalid type", instr.pos)
            self.check_dst(instr, instr.type)
            self.check_value(instr.value, "cast source")
            return

        if isinstance(instr, IRMathOp):
            self.verify_math(instr)
            return

        if isinstance(instr, IRVecReduce):
            self.check_dst(instr, instr.type)
            self.check_value(instr.vec, "reduction operand")
            if not is_vector(instr.vec.type):
                self.err(f"reduction operand has type '{ts(instr.vec.type)}', "
                         f"expected a vector", instr.pos)
            if instr.type != vector_elem(instr.vec.type):
                self.err(f"reduction result has type '{ts(instr.type)}', "
                         f"expected '{ts(vector_elem(instr.vec.type))}'",
                         instr.pos)
            if instr.op not in ('sum', 'max', 'min'):
                self.err(f"unknown reduction '{instr.op}'", instr.pos)
            return

        if isinstance(instr, IRSplat):
            self.check_dst(instr, instr.type)
            if not is_vector(instr.type):
                self.err(f"splat result has type '{ts(instr.type)}', "
                         f"expected a vector", instr.pos)
            self.check_assignable(instr.value, vector_elem(instr.type),
                                  "splat value")
            return

        if isinstance(instr, IRVecBinOp):
            self.verify_vec_binop(instr)
            return

        if isinstance(instr, IRArrayLit):
            self.verify_array_lit(instr)
            return

        if isinstance(instr, IRArrayIndex):
            self.verify_array_index(instr)
            return

        if isinstance(instr, IRArraySet):
            self.verify_array_set(instr)
            return

        if isinstance(instr, IRRef):
            self.check_dst(instr, instr.type)
            if not isinstance(instr.value, IRLocal):
                self.err("reference target must be a named local", instr.pos)
            self.check_value(instr.value, "reference target")
            if instr.type != ('ptr', instr.value.type):
                self.err(f"reference result has type '{ts(instr.type)}', "
                         f"expected '{ts(('ptr', instr.value.type))}'",
                         instr.pos)
            return

        if isinstance(instr, IRAlloc):
            if not is_array(instr.array_type):
                self.err(f"alloc type '{ts(instr.array_type)}' is not an array",
                         instr.pos)
            self.check_dst(instr, ('ptr', instr.array_type))
            return

        if isinstance(instr, IRStructInit):
            self.verify_struct_init(instr)
            return

        if isinstance(instr, IRFieldGet):
            self.check_dst(instr, instr.type)
            self.check_value(instr.base, "field base")
            if not _is_struct(self.module, instr.base.type):
                self.err(f"field base has type '{ts(instr.base.type)}', "
                         f"expected a struct", instr.pos)
            ftype = _field_type(self.module, instr.base.type[1], instr.field)
            if ftype is None:
                self.err(f"struct '{instr.base.type[1]}' has no field "
                         f"'{instr.field}'", instr.pos)
            if instr.type != ftype:
                self.err(f"field '{instr.field}' has type '{ts(ftype)}', "
                         f"instruction says '{ts(instr.type)}'", instr.pos)
            return

        if isinstance(instr, IRVecLit):
            self.verify_vec_lit(instr)
            return

        if isinstance(instr, IRVecIndex):
            self.check_dst(instr, instr.type)
            self.check_value(instr.vec, "vector index base")
            if not is_vector(instr.vec.type):
                self.err(f"vector index base has type '{ts(instr.vec.type)}'",
                         instr.pos)
            if instr.type != vector_elem(instr.vec.type):
                self.err(f"vector index result has type '{ts(instr.type)}', "
                         f"expected '{ts(vector_elem(instr.vec.type))}'",
                         instr.pos)
            self.check_index(instr.index)
            return

        if isinstance(instr, IRVecSet):
            self.check_value(instr.vec, "vector set base")
            if not is_vector(instr.vec.type):
                self.err(f"vector set base has type '{ts(instr.vec.type)}'",
                         instr.pos)
            if not isinstance(instr.vec, IRLocal):
                self.err("vector set base must be a named local", instr.pos)
            self.check_assignable(instr.value, vector_elem(instr.vec.type),
                                  "vector set value")
            self.check_index(instr.index)
            return

        if isinstance(instr, IRReturn):
            ret = self.func.ret
            if self.func.name == 'main' and _is_void(ret):
                ret = I32
            if _is_void(ret):
                if instr.value is not None:
                    self.err("void function returns a value", instr.pos)
            else:
                self.check_assignable(instr.value, ret, "return value")
            return

        if isinstance(instr, IRJump):
            if instr.target not in self.labels:
                self.err(f"jump to undefined block '{instr.target}'",
                         instr.pos)
            return

        if isinstance(instr, IRCondJump):
            self.check_value(instr.cond, "condition")
            ct = instr.cond.type
            if ct != BOOL:
                self.err(f"condition has type '{ts(ct)}', expected bool",
                         instr.pos)
            if instr.then_target not in self.labels:
                self.err(f"conditional jump to undefined block "
                         f"'{instr.then_target}'", instr.pos)
            if instr.else_target not in self.labels:
                self.err(f"conditional jump to undefined block "
                         f"'{instr.else_target}'", instr.pos)
            return

        self.err(f"unsupported instruction {type(instr).__name__}",
                 instr.pos)

    # -- specific instruction families -------------------------------------
    def verify_binop(self, instr):
        self.check_dst(instr, instr.type)
        self.check_value(instr.left, "left operand")
        self.check_value(instr.right, "right operand")
        lt, rt, out = instr.left.type, instr.right.type, instr.type
        op = instr.op

        if is_vector(out):
            if op not in ('+', '-', '*', '/'):
                self.err(f"vector operation '{op}' is not supported",
                         instr.pos)
            if lt != out or rt != out:
                self.err(f"vector operation operands have types "
                         f"'{ts(lt)}' and '{ts(rt)}', expected '{ts(out)}'",
                         instr.pos)
            return

        if op in ('==', '!=', '<', '<=', '>', '>='):
            if out != BOOL:
                self.err(f"comparison result has type '{ts(out)}', "
                         f"expected bool", instr.pos)
            if is_vector(lt) or is_vector(rt):
                self.err("vector comparison is not supported", instr.pos)
            if is_array(lt) or is_array(rt):
                self.err(f"cannot compare arrays '{ts(lt)}' and '{ts(rt)}'",
                         instr.pos)
            if lt == STR or rt == STR:
                self.err("string comparison is not supported", instr.pos)
            if lt == BOOL or rt == BOOL:
                if lt != rt:
                    self.err(f"cannot compare '{ts(lt)}' and '{ts(rt)}'",
                             instr.pos)
                if op not in ('==', '!='):
                    self.err("bool values only support '==' and '!='",
                             instr.pos)
                return
            if lt[0] == 'named' or rt[0] == 'named':
                if lt != rt:
                    self.err(f"cannot compare '{ts(lt)}' and '{ts(rt)}'",
                             instr.pos)
                if lt[1] in self.module.structs:
                    self.err(f"cannot compare structs '{lt[1]}'", instr.pos)
                if lt[1] in self.module.enums:
                    return
                self.err(f"cannot compare '{ts(lt)}'", instr.pos)
            if lt[0] == 'ptr' or rt[0] == 'ptr':
                if lt != rt:
                    self.err(f"cannot compare different pointer types "
                             f"'{ts(lt)}' and '{ts(rt)}'", instr.pos)
                return
            if not (is_num(lt) and is_num(rt)):
                self.err(f"cannot compare '{ts(lt)}' and '{ts(rt)}'",
                         instr.pos)
            return

        if op in ('+', '-') and (_is_pointer(lt) or _is_pointer(rt)):
            if op == '-' and _is_pointer(lt) and _is_pointer(rt):
                if lt != rt:
                    self.err(f"pointer subtraction between '{ts(lt)}' and "
                             f"'{ts(rt)}'", instr.pos)
                if out != I64:
                    self.err(f"pointer subtraction result has type "
                             f"'{ts(out)}', expected i64", instr.pos)
                return
            if _is_pointer(lt):
                if lt != out or not is_int(rt):
                    self.err(f"pointer arithmetic between '{ts(lt)}' and "
                             f"'{ts(rt)}'", instr.pos)
                return
            if _is_pointer(rt):
                if op != '+' or rt != out or not is_int(lt):
                    self.err(f"pointer arithmetic between '{ts(lt)}' and "
                             f"'{ts(rt)}'", instr.pos)
                return

        if op in ('%', '<<', '>>', '&', '|', '^'):
            if not ((is_int(lt) or is_quant(lt))
                    and (is_int(rt) or is_quant(rt))):
                self.err(f"integer operation '{op}' on '{ts(lt)}' and "
                         f"'{ts(rt)}'", instr.pos)
        elif not (is_num(lt) and is_num(rt)):
            self.err(f"arithmetic operation '{op}' on '{ts(lt)}' and "
                     f"'{ts(rt)}'", instr.pos)

        if not is_num(out):
            self.err(f"arithmetic result has type '{ts(out)}'", instr.pos)

    def verify_unop(self, instr):
        self.check_dst(instr, instr.type)
        self.check_value(instr.operand, "unary operand")
        ot = instr.operand.type
        if instr.op == '-':
            if not (is_num(ot) or is_vector(ot)):
                self.err(f"unary '-' on '{ts(ot)}'", instr.pos)
        elif instr.op == '!':
            if ot != BOOL:
                self.err(f"unary '!' requires a bool operand, got "
                         f"'{ts(ot)}'", instr.pos)
        elif instr.op == '~':
            if not (is_int(ot) or is_quant(ot)):
                self.err(f"unary '~' on '{ts(ot)}'", instr.pos)
        else:
            self.err(f"unknown unary operator '{instr.op}'", instr.pos)
        if instr.type != ot:
            self.err(f"unary result has type '{ts(instr.type)}', "
                     f"expected '{ts(ot)}'", instr.pos)

    def verify_call(self, instr):
        if instr.type is None:
            self.err("call has no result type", instr.pos)
        if instr.type == VOID:
            if instr.dst is not None:
                self.err("void call has a destination", instr.pos)
        else:
            self.check_dst(instr, instr.type)

        for i, arg in enumerate(instr.args, 1):
            self.check_value(arg, f"argument {i}")

        target = self.resolve_callee(instr.callee)
        if target is None:
            self.err(f"call to unknown callee '{instr.callee}'", instr.pos)
        params, ret = target
        if len(instr.args) != len(params):
            self.err(f"'{instr.callee}' expects {len(params)} argument(s), "
                     f"got {len(instr.args)}", instr.pos)
        for i, (arg, ptype) in enumerate(zip(instr.args, params), 1):
            if ptype is ANY_PTR:
                if not _is_pointer(arg.type):
                    self.err(f"argument {i} to '{instr.callee}' has type "
                             f"'{ts(arg.type)}', expected a pointer",
                             instr.pos)
            elif arg.type != ptype:
                self.err(f"argument {i} to '{instr.callee}' has type "
                         f"'{ts(arg.type)}', expected '{ts(ptype)}'",
                         instr.pos)
        if ret is ANY_PTR:
            expected = instr.type
        else:
            expected = ret if ret is not None else VOID
        if instr.type != expected:
            self.err(f"call result has type '{ts(instr.type)}', "
                     f"expected '{ts(expected)}'", instr.pos)

    def resolve_callee(self, callee):
        for f in self.module.funcs:
            if f.cname == callee:
                return [ptype for _, ptype in f.params], f.ret
        if callee in self.module.externs:
            params, ret = self.module.externs[callee]
            return [ptype for _, ptype in params], ret
        if callee in RUNTIME_CALLS:
            return RUNTIME_CALLS[callee]
        return None

    def verify_math(self, instr):
        self.check_dst(instr, instr.type)
        if len(instr.args) != 1:
            self.err(f"{instr.op} expects one argument", instr.pos)
        self.check_value(instr.args[0], "math operand")
        at = instr.args[0].type
        if instr.op not in ('sqrt', 'rsqrt', 'exp', 'tanh', 'erf',
                            'cos', 'sin'):
            self.err(f"unknown math operation '{instr.op}'", instr.pos)
        if is_vector(instr.type):
            if instr.type != ('prim', 'v4f32') or at != instr.type:
                self.err(f"{instr.op} vector types '{ts(at)}' -> "
                         f"'{ts(instr.type)}'", instr.pos)
        elif instr.type not in (('prim', 'f32'), ('prim', 'f64')):
            self.err(f"{instr.op} result has type '{ts(instr.type)}'",
                     instr.pos)
        elif at != instr.type:
            self.err(f"{instr.op} operand has type '{ts(at)}', expected "
                     f"'{ts(instr.type)}'", instr.pos)

    def verify_vec_binop(self, instr):
        self.check_dst(instr, instr.type)
        if not is_vector(instr.type):
            self.err(f"vector operation result has type '{ts(instr.type)}'",
                     instr.pos)
        expected = 3 if instr.op == 'vselect' else 2
        if instr.op not in ('vmax2', 'vmin2', 'vcmplt', 'vcmpgt',
                            'vselect'):
            self.err(f"unknown vector operation '{instr.op}'", instr.pos)
        if len(instr.args) != expected:
            self.err(f"{instr.op} expects {expected} argument(s)",
                     instr.pos)
        for i, arg in enumerate(instr.args, 1):
            self.check_assignable(arg, instr.type, f"argument {i} to "
                                  f"{instr.op}")

    def verify_array_lit(self, instr):
        self.check_dst(instr, instr.type)
        if not is_array(instr.type):
            self.err(f"array literal has non-array type '{ts(instr.type)}'",
                     instr.pos)
        count = array_len(instr.type)
        if len(instr.elems) != count:
            self.err(f"array literal has {len(instr.elems)} element(s), "
                     f"expected {count}", instr.pos)
        for i, elem in enumerate(instr.elems, 1):
            self.check_assignable(elem, array_elem(instr.type),
                                  f"array element {i}")

    def verify_array_index(self, instr):
        self.check_dst(instr, instr.type)
        self.check_value(instr.base, "array index base")
        elem = _array_index_elem(instr.base.type)
        if elem is None:
            self.err(f"cannot index '{ts(instr.base.type)}'", instr.pos)
        if instr.type != elem:
            self.err(f"array index result has type '{ts(instr.type)}', "
                     f"expected '{ts(elem)}'", instr.pos)
        self.check_index(instr.index)

    def verify_array_set(self, instr):
        self.check_value(instr.base, "array set base")
        elem = _array_index_elem(instr.base.type)
        if elem is None:
            self.err(f"cannot index-assign '{ts(instr.base.type)}'",
                     instr.pos)
        self.check_assignable(instr.value, elem, "array set value")
        self.check_index(instr.index)

    def verify_struct_init(self, instr):
        self.check_dst(instr, instr.type)
        if not _is_struct(self.module, instr.type):
            self.err(f"struct initializer has type '{ts(instr.type)}'",
                     instr.pos)
        seen = set()
        for fname, value in instr.fields:
            if fname in seen:
                self.err(f"duplicate field '{fname}'", instr.pos)
            seen.add(fname)
            ftype = _field_type(self.module, instr.type[1], fname)
            if ftype is None:
                self.err(f"struct '{instr.type[1]}' has no field '{fname}'",
                         instr.pos)
            self.check_assignable(value, ftype, f"field '{fname}'")

    def verify_vec_lit(self, instr):
        self.check_dst(instr, instr.type)
        if not is_vector(instr.type):
            self.err(f"vector literal has non-vector type "
                     f"'{ts(instr.type)}'", instr.pos)
        count = vector_len(instr.type)
        if len(instr.elems) != count:
            self.err(f"vector literal has {len(instr.elems)} element(s), "
                     f"expected {count}", instr.pos)
        elem = vector_elem(instr.type)
        for i, value in enumerate(instr.elems, 1):
            self.check_assignable(value, elem, f"vector element {i}")


def verify_module(module, fname="<ir>"):
    """Verify a lowered module. Returns IRStats or raises CompileError."""
    return Verifier(module, fname).verify()
