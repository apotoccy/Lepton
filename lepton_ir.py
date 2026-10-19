"""
Lepton intermediate representation.

This is the layer between the AST and any backend. Right now it exists only
to prove the pipeline on a minimal program. Every language feature added
from here on lands as an IR construct first, then gets a lowering rule and
an emission rule per backend.

Design notes
------------
* Every instruction carries a `pos` tuple (line, col) copied from the AST
  node it was lowered from, so backend errors can point at source.
* Operands are `IRTemp`, `IRLocal`, or `IRConst`. Temps are unique per
  function and rendered as `_t0`, `_t1`, ... in C. Locals are named after
  the source variable. Constants are inline.
* A block is a list of instructions. Terminators (`IRReturn`, `IRJump`,
  `IRCondJump`) are ordinary instructions that happen to end the block.
* Types are the same tuples the AST uses: `('prim', 'i32')`,
  `('ptr', ...)`, `('named', 'Point')`, and so on. When the type system
  gets more complex (Phase 1), the IR will keep pace.
"""

from __future__ import annotations


class IRNotImplemented(Exception):
    """Raised by the lowerer for constructs the IR pipeline does not yet
    handle. Distinct from a real compile error, so the test harness can
    treat unsupported cases as skips rather than failures."""
    pass


# ---------------------------------------------------------------------------
#  Values (operands)
# ---------------------------------------------------------------------------

class IRValue:
    """Base class for anything that can appear as an operand."""
    type = None


class IRTemp(IRValue):
    """A compiler-generated SSA value, unique within its function."""
    def __init__(self, id, type):
        self.id = id
        self.type = type

    def __repr__(self):
        return f"%{self.id}"


class IRLocal(IRValue):
    """A named source-level variable (parameter, let, or var)."""
    def __init__(self, name, type):
        self.name = name
        self.type = type

    def __repr__(self):
        return f"${self.name}"


class IRConst(IRValue):
    """A literal constant. `value` is the Python value itself."""
    def __init__(self, value, type):
        self.value = value
        self.type = type

    def __repr__(self):
        return f"const({self.value!r})"


# ---------------------------------------------------------------------------
#  Instructions
# ---------------------------------------------------------------------------

class IRInstr:
    """Base class. Every instruction has a source position."""

    def __init__(self, pos=None):
        self.pos = pos

    def operands(self):
        """Return all IRValue operands (for rewriting passes)."""
        return []

    def replace_operand(self, old, new):
        """Replace one operand with another. Subclasses override."""
        pass


class IRBind(IRInstr):
    """Bind a value to a new named local. Corresponds to `let` / `var`."""

    def __init__(self, name, value, type, mutable, pos=None):
        super().__init__(pos)
        self.name = name
        self.value = value
        self.type = type
        self.mutable = mutable

    def operands(self):
        return [self.value]

    def replace_operand(self, old, new):
        if self.value is old:
            self.value = new


class IRAssign(IRInstr):
    """Assign a new value to an existing local."""

    def __init__(self, name, value, pos=None):
        super().__init__(pos)
        self.name = name
        self.value = value

    def operands(self):
        return [self.value]

    def replace_operand(self, old, new):
        if self.value is old:
            self.value = new


class IRBinOp(IRInstr):
    """dst = left op right."""

    def __init__(self, dst, op, left, right, type, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.op = op
        self.left = left
        self.right = right
        self.type = type

    def operands(self):
        return [self.left, self.right]

    def replace_operand(self, old, new):
        if self.left is old: self.left = new
        if self.right is old: self.right = new


class IRUnOp(IRInstr):
    """dst = op operand."""

    def __init__(self, dst, op, operand, type, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.op = op
        self.operand = operand
        self.type = type

    def operands(self):
        return [self.operand]

    def replace_operand(self, old, new):
        if self.operand is old:
            self.operand = new


class IRCall(IRInstr):
    """dst = callee(args...). `dst` is None for void calls.

    `callee` is the C-level name (e.g. "lp_println", "lp_add", "main").
    The lowerer resolves source names to C names, so the backend does not
    need to know about `lp_` prefixes or builtins.
    """

    def __init__(self, dst, callee, args, type, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.callee = callee
        self.args = args
        self.type = type

    def operands(self):
        return list(self.args)

    def replace_operand(self, old, new):
        self.args = [new if a is old else a for a in self.args]

class IRConcat(IRInstr):
    """dst = lp_cat(left, right). Both operands are str-typed."""
    def __init__(self, dst, left, right, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.left = left
        self.right = right
        self.type = ('prim', 'str')

    def operands(self):
        return [self.left, self.right]

    def replace_operand(self, old, new):
        if self.left is old: self.left = new
        if self.right is old: self.right = new


class IRToString(IRInstr):
    """dst = string form of `value`. `src_type` is value.type, kept on the
    instruction for the emitter's dispatch. dst is always str-typed."""
    def __init__(self, dst, value, src_type, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.value = value
        self.src_type = src_type
        self.type = ('prim', 'str')

    def operands(self):
        return [self.value]

    def replace_operand(self, old, new):
        if self.value is old:
            self.value = new

class IRCast(IRInstr):
    """dst = (type) value.

    `type` is the target type. `value.type` is the source type. The
    semantic rules for which casts are legal live in the lowerer, not
    here — the IR just records what the source asked for.
    """
    def __init__(self, dst, value, type, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.value = value
        self.type = type

    def operands(self):
        return [self.value]

    def replace_operand(self, old, new):
        if self.value is old:
            self.value = new
         
class IRMathOp(IRInstr):
    """dst = op(arg). op is one of: sqrt, rsqrt, exp, tanh, erf.

    For scalars, `type` is the result type and args[0] is the operand.
    For vectors, both are vector types. The dispatch to a scalar helper
    or a target intrinsic happens at emission time.
    """
    def __init__(self, dst, op, args, type, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.op = op
        self.args = args
        self.type = type

    def operands(self):
        return list(self.args)

    def replace_operand(self, old, new):
        self.args = [new if a is old else a for a in self.args]
   
class IRVecReduce(IRInstr):
    """dst = reduce_op(vec). op is one of: sum, max, min.

    `vec` is a vector-typed IRValue; `type` is the scalar element type
    of that vector. The reduction collapses the vector to a single value
    of the element type.
    """
    def __init__(self, dst, op, vec, type, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.op = op
        self.vec = vec
        self.type = type

    def operands(self):
        return [self.vec]

    def replace_operand(self, old, new):
        if self.vec is old:
            self.vec = new
            
class IRSplat(IRInstr):
    """dst = broadcast of scalar `value` to a vector of type `type`.

    The value's type is the vector's element type. Used by the lowerer
    to promote a scalar operand when the other side of a binary op is a
    vector.
    """
    def __init__(self, dst, value, type, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.value = value
        self.type = type

    def operands(self):
        return [self.value]

    def replace_operand(self, old, new):
        if self.value is old:
            self.value = new
            
class IRVecBinOp(IRInstr):
    """dst = vec_op(args...).

    op is one of: vmax2, vmin2, vcmplt, vcmpgt, vselect.

    For vmax2 / vmin2 / vcmplt / vcmpgt, args is [a, b] with both
    vectors of the same type, and `type` is that type.

    For vselect, args is [mask, then_value, else_value]. The mask is a
    vector of the same shape as the values; a lane of the mask that has
    all bits set selects from `then`, otherwise from `else`. `type` is
    the value type.
    """
    def __init__(self, dst, op, args, type, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.op = op
        self.args = args
        self.type = type

    def operands(self):
        return list(self.args)

    def replace_operand(self, old, new):
        self.args = [new if a is old else a for a in self.args]
        
class IRArrayLit(IRInstr):
    """dst = { e0, e1, ..., eN } of the given array type."""
    def __init__(self, dst, elems, type, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.elems = elems
        self.type = type

    def operands(self):
        return list(self.elems)

    def replace_operand(self, old, new):
        self.elems = [new if e is old else e for e in self.elems]


class IRArrayIndex(IRInstr):
    """dst = base[index]."""
    def __init__(self, dst, base, index, type, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.base = base
        self.index = index
        self.type = type

    def operands(self):
        return [self.base, self.index]

    def replace_operand(self, old, new):
        if self.base is old:  self.base = new
        if self.index is old: self.index = new


class IRArraySet(IRInstr):
    """base[index] = value. A statement, not an expression: no dst."""
    def __init__(self, base, index, value, pos=None):
        super().__init__(pos)
        self.base = base
        self.index = index
        self.value = value

    def operands(self):
        return [self.base, self.index, self.value]

    def replace_operand(self, old, new):
        if self.base is old:  self.base = new
        if self.index is old: self.index = new
        if self.value is old: self.value = new        
            
class IRRef(IRInstr):
    """dst = &value. Only a nameable local can be referenced.

    `value` is an IRLocal of the referent type; `type` is ('ptr', T).
    """
    def __init__(self, dst, value, type, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.value = value
        self.type = type

    def operands(self):
        return [self.value]

    def replace_operand(self, old, new):
        if self.value is old:
            self.value = new

class IRAlloc(IRInstr):
    """dst = heap allocation of one array of the given type.

    `array_type` is the ('array', elem, count) type; `type` is
    ('ptr', array_type). The pointer is uninitialized — the caller
    is expected to write before reading, or to zero the array first.
    """
    def __init__(self, dst, array_type, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.array_type = array_type
        self.type = ('ptr', array_type)

    def operands(self):
        return []
            
class IRVecLit(IRInstr):
    """dst = { e0, e1, ..., eN } of the given vector type."""
    def __init__(self, dst, elems, type, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.elems = elems
        self.type = type

    def operands(self):
        return list(self.elems)

    def replace_operand(self, old, new):
        self.elems = [new if e is old else e for e in self.elems]


class IRVecIndex(IRInstr):
    """dst = vec[index]. index is any integer-valued IRValue."""
    def __init__(self, dst, vec, index, type, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.vec = vec
        self.index = index
        self.type = type

    def operands(self):
        return [self.vec, self.index]

    def replace_operand(self, old, new):
        if self.vec is old:   self.vec = new
        if self.index is old: self.index = new

class IRVecSet(IRInstr):
    """vec[index] = value. A statement: no dst.

    Assigns one lane of a vector. `value`'s type is the vector's element
    type. The vector must be a nameable local; `f()[i] = x` is not
    supported.
    """
    def __init__(self, vec, index, value, pos=None):
        super().__init__(pos)
        self.vec = vec
        self.index = index
        self.value = value

    def operands(self):
        return [self.vec, self.index, self.value]

    def replace_operand(self, old, new):
        if self.vec is old:   self.vec = new
        if self.index is old: self.index = new
        if self.value is old: self.value = new

class IRStructInit(IRInstr):
    """dst = StructName{ .field = value, ... }.

    `fields` is a list of (field_name, IRValue) in declaration order, with
    only the fields the source actually supplied. Missing fields default
    to zero, matching C's designated-initializer semantics.
    """

    def __init__(self, dst, fields, type, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.fields = fields
        self.type = type

    def operands(self):
        return [v for _, v in self.fields]

    def replace_operand(self, old, new):
        self.fields = [(n, new if v is old else v) for n, v in self.fields]


class IRFieldGet(IRInstr):
    """dst = base.field.

    `base` is any struct-typed IRValue; `field` is the field name; `type`
    is the field's declared type.
    """

    def __init__(self, dst, base, field, type, pos=None):
        super().__init__(pos)
        self.dst = dst
        self.base = base
        self.field = field
        self.type = type

    def operands(self):
        return [self.base]

    def replace_operand(self, old, new):
        if self.base is old:
            self.base = new

class IRReturn(IRInstr):
    """Return from the current function. `value` is None for void returns."""

    def __init__(self, value, pos=None):
        super().__init__(pos)
        self.value = value

    def operands(self):
        return [self.value] if self.value is not None else []

    def replace_operand(self, old, new):
        if self.value is old:
            self.value = new


class IRJump(IRInstr):
    """Unconditional jump to a block label."""

    def __init__(self, target, pos=None):
        super().__init__(pos)
        self.target = target


class IRCondJump(IRInstr):
    """Conditional jump: if cond then `then_target` else `else_target`."""

    def __init__(self, cond, then_target, else_target, pos=None):
        super().__init__(pos)
        self.cond = cond
        self.then_target = then_target
        self.else_target = else_target

    def operands(self):
        return [self.cond]

    def replace_operand(self, old, new):
        if self.cond is old:
            self.cond = new


TERMINATORS = (IRReturn, IRJump, IRCondJump)


# ---------------------------------------------------------------------------
#  Blocks, functions, modules
# ---------------------------------------------------------------------------

class IRBlock:
    """A basic block: label, instructions, optional terminator."""

    def __init__(self, label, pos=None):
        self.label = label
        self.pos = pos
        self.instrs = []

    def add(self, instr):
        self.instrs.append(instr)
        return instr

    def is_terminated(self):
        return bool(self.instrs) and isinstance(self.instrs[-1], TERMINATORS)

    def __repr__(self):
        return f"<IRBlock {self.label} ({len(self.instrs)} instrs)>"


class IRFunc:
    """A function: parameters, return type, and a list of blocks."""

    def __init__(self, name, params, ret, pos=None):
        self.name = name
        self.cname = None              # C symbol (set by the lowerer)
        self.params = params           # list of (name, type)
        self.ret = ret                 # lepton type tuple, or None for void
        self.pos = pos
        self.blocks = []
        self.entry = None
        self.temp_count = 0
        self.temp_count = 0
        self.label_count = 0

    def new_temp(self, type=None):
        t = IRTemp(self.temp_count, type)
        self.temp_count += 1
        return t
        
    def new_label(self, base):
        self.label_count += 1
        return f"{base}_{self.label_count}"

    def add_block(self, label, pos=None):
        b = IRBlock(label, pos)
        self.blocks.append(b)
        if self.entry is None:
            self.entry = b
        return b

    def __repr__(self):
        return f"<IRFunc {self.name} ({len(self.blocks)} blocks)>"


class IRModule:
    """Top-level IR container: functions, nominal types, globals, externs."""

    def __init__(self, name="module"):
        self.name = name
        self.funcs = []
        self.structs = {}      # name -> list of (field_name, type)
        self.enums = {}        # name -> list of variant names
        self.globals = {}      # name -> (cname, type)
        self.externs = {}      # name -> (list of (name, type), ret)

# ---------------------------------------------------------------------------
#  IR pretty-printer
# ---------------------------------------------------------------------------

def _fmt_type(t):
    if t is None:
        return 'void'
    if t[0] == 'prim':
        return t[1]
    if t[0] == 'ptr':
        return f"ptr[{_fmt_type(t[1])}]"
    if t[0] == 'array':
        return f"[{_fmt_type(t[1])}; {t[2]}]"
    if t[0] == 'named':
        return t[1]
    return str(t)


def render_value(v):
    if isinstance(v, IRTemp):
        return f"%{v.id}"
    if isinstance(v, IRLocal):
        return f"${v.name}"
    if isinstance(v, IRConst):
        val = v.value
        if isinstance(val, str):
            return f'"{val}"'
        if isinstance(val, bool):
            return 'true' if val else 'false'
        return repr(val)
    return f"<{v!r}>"


def render_instr(instr):
    if isinstance(instr, IRBind):
        kw = 'var' if instr.mutable else 'let'
        return f"{kw} {instr.name} = {render_value(instr.value)}"
    if isinstance(instr, IRAssign):
        return f"{instr.name} = {render_value(instr.value)}"
    if isinstance(instr, IRBinOp):
        return (f"%{instr.dst.id} = {render_value(instr.left)} {instr.op} "
                f"{render_value(instr.right)}")
    if isinstance(instr, IRUnOp):
        return f"%{instr.dst.id} = {instr.op}{render_value(instr.operand)}"
    if isinstance(instr, IRCall):
        args = ", ".join(render_value(a) for a in instr.args)
        lhs = f"%{instr.dst.id} = " if instr.dst else ""
        return f"{lhs}{instr.callee}({args})"
    if isinstance(instr, IRReturn):
        if instr.value is None:
            return "return"
        return f"return {render_value(instr.value)}"
    if isinstance(instr, IRJump):
        return f"goto {instr.target}"
    if isinstance(instr, IRCondJump):
        return (f"branch {render_value(instr.cond)} "
                f"? {instr.then_target} : {instr.else_target}")
    if isinstance(instr, IRConcat):
        return (f"%{instr.dst.id} = concat("
                f"{render_value(instr.left)}, {render_value(instr.right)})")
    if isinstance(instr, IRToString):
        return (f"%{instr.dst.id} = str({render_value(instr.value)})")
    if isinstance(instr, IRCast):
        return (f"%{instr.dst.id} = cast<{_fmt_type(instr.type)}>("
                f"{render_value(instr.value)})")
    if isinstance(instr, IRMathOp):
        args = ", ".join(render_value(a) for a in instr.args)
        return f"%{instr.dst.id} = {instr.op}({args})"
    if isinstance(instr, IRVecReduce):
        return f"%{instr.dst.id} = reduce_{instr.op}({render_value(instr.vec)})"
    if isinstance(instr, IRSplat):
        return f"%{instr.dst.id} = splat({render_value(instr.value)})"
    if isinstance(instr, IRVecBinOp):
        args = ", ".join(render_value(a) for a in instr.args)
        return f"%{instr.dst.id} = {instr.op}({args})"
    if isinstance(instr, IRStructInit):
        parts = ", ".join(f".{n}={render_value(v)}"
                          for n, v in instr.fields) or "0"
        return f"%{instr.dst.id} = {instr.type[1]}{{ {parts} }}"
    if isinstance(instr, IRFieldGet):
        return (f"%{instr.dst.id} = "
                f"{render_value(instr.base)}.{instr.field}")
    if isinstance(instr, IRVecLit):
        elems = ", ".join(render_value(e) for e in instr.elems)
        return f"%{instr.dst.id} = vec[ {elems} ]"
    if isinstance(instr, IRVecIndex):
        return (f"%{instr.dst.id} = "
                f"{render_value(instr.vec)}[{render_value(instr.index)}]")
    if isinstance(instr, IRVecSet):
        return (f"{render_value(instr.vec)}[{render_value(instr.index)}] = "
                f"{render_value(instr.value)}")
    if isinstance(instr, IRArrayLit):
        elems = ", ".join(render_value(e) for e in instr.elems)
        return f"%{instr.dst.id} = arr{{ {elems} }}"
    if isinstance(instr, IRArrayIndex):
        return (f"%{instr.dst.id} = "
                f"{render_value(instr.base)}[{render_value(instr.index)}]")
    if isinstance(instr, IRArraySet):
        return (f"{render_value(instr.base)}"
                f"[{render_value(instr.index)}] = {render_value(instr.value)}")
    if isinstance(instr, IRRef):
        return f"%{instr.dst.id} = &{render_value(instr.value)}"
    if isinstance(instr, IRAlloc):
        return (f"%{instr.dst.id} = alloc<{_fmt_type(instr.array_type)}>()")
    return f"<unknown {type(instr).__name__}>"


def render_module(module):
    lines = []

    for name, fields in module.structs.items():
        parts = ", ".join(f"{f}: {_fmt_type(ft)}" for f, ft in fields)
        lines.append(f"struct {name} {{ {parts} }}")
    for name, variants in module.enums.items():
        lines.append(f"enum {name} {{ {', '.join(variants)} }}")
    if module.structs or module.enums:
        lines.append("")

    for f in module.funcs:
        params = ", ".join(f"{n}: {_fmt_type(t)}" for n, t in f.params)
        ret = _fmt_type(f.ret) if f.ret else 'void'
        lines.append(f"fn {f.name}({params}) -> {ret} {{")
        for block in f.blocks:
            lines.append(f"  {block.label}:")
            for instr in block.instrs:
                lines.append(f"    {render_instr(instr)}")
        lines.append("}")
        lines.append("")

    return '\n'.join(lines)