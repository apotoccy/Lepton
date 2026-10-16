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
