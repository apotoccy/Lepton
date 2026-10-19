#!/usr/bin/env python3
"""Unit tests for the IR verifier.

These construct IR directly (no source lowering) so that deliberately
malformed modules can be fed through the verifier.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lepton_ir import (
    IRModule, IRFunc, IRTemp, IRLocal, IRConst,
    IRBind, IRReturn, IRJump, IRCall, IRArrayLit, IRStructInit,
)
from lepton_verify import verify_module
from leptonc import CompileError


I32 = ('prim', 'i32')
F32 = ('prim', 'f32')
STR = ('prim', 'str')
VOID = ('prim', 'void')


def new_func(name='f', ret=VOID):
    f = IRFunc(name, [], ret)
    f.cname = 'main' if name == 'main' else 'lp_' + name
    return f


def new_module(*funcs):
    m = IRModule()
    m.funcs.extend(funcs)
    return m


def expect_ok(module):
    verify_module(module, '<test>')


def expect_error(module, needle):
    try:
        verify_module(module, '<test>')
    except CompileError as e:
        text = str(e)
        if needle not in text:
            raise AssertionError(
                f"expected {needle!r} in verifier error, got: {text}")
        return
    raise AssertionError(f"expected verifier error containing {needle!r}")


def test_valid_module():
    f = new_func('main', I32)
    b = f.add_block('entry')
    b.add(IRReturn(IRConst(0, I32)))
    m = new_module(f)
    stats = verify_module(m, '<test>')
    assert (stats.functions, stats.blocks, stats.instructions) == (1, 1, 1), stats


def test_bind_type_mismatch():
    f = new_func()
    b = f.add_block('entry')
    b.add(IRBind('x', IRConst(1, I32), F32, False))
    b.add(IRReturn(None))
    expect_error(new_module(f), "initializer for 'x'")


def test_undefined_jump_target():
    f = new_func()
    b = f.add_block('entry')
    b.add(IRJump('missing'))
    expect_error(new_module(f), "jump to undefined block 'missing'")


def test_call_arity():
    f = new_func()
    b = f.add_block('entry')
    b.add(IRCall(None, 'lp_print', [], VOID))
    b.add(IRReturn(None))
    expect_error(new_module(f), "expects 1 argument(s), got 0")


def test_return_type_mismatch():
    f = new_func('f', I32)
    b = f.add_block('entry')
    b.add(IRReturn(IRConst("hi", STR)))
    expect_error(new_module(f), "return value has type 'str'")


def test_duplicate_block_label():
    f = new_func()
    b1 = f.add_block('entry')
    b1.add(IRReturn(None))
    b2 = f.add_block('entry')
    b2.add(IRReturn(None))
    expect_error(new_module(f), "duplicate block label 'entry'")


def test_undefined_temp():
    f = new_func('f', I32)
    b = f.add_block('entry')
    b.add(IRReturn(IRTemp(99, I32)))
    expect_error(new_module(f), "undefined temporary %99")


def test_undefined_local():
    f = new_func('f', I32)
    b = f.add_block('entry')
    b.add(IRReturn(IRLocal('x', I32)))
    expect_error(new_module(f), "undefined local 'x'")


def test_array_literal_count():
    arr = ('array', I32, 2)
    f = new_func('f', arr)
    b = f.add_block('entry')
    tmp = f.new_temp(arr)
    b.add(IRArrayLit(tmp, [IRConst(1, I32)], arr))
    b.add(IRReturn(tmp))
    expect_error(new_module(f), "array literal has 1 element(s), expected 2")


def test_struct_field_type():
    m = IRModule()
    m.structs['P'] = [('x', I32)]
    f = new_func()
    b = f.add_block('entry')
    tmp = f.new_temp(('named', 'P'))
    b.add(IRStructInit(tmp, [('x', IRConst("hi", STR))], ('named', 'P')))
    b.add(IRReturn(None))
    m.funcs.append(f)
    expect_error(m, "field 'x' has type 'str'")


def test_reachable_fallthrough():
    f = new_func('f', I32)
    b = f.add_block('entry')
    b.add(IRBind('x', IRConst(1, I32), I32, False))
    expect_error(new_module(f), "falls through without a return")


def test_instructions_after_terminator():
    f = new_func()
    b = f.add_block('entry')
    b.add(IRReturn(None))
    b.add(IRBind('x', IRConst(1, I32), I32, False))
    expect_error(new_module(f), "instructions after a terminator")


def test_unknown_callee():
    f = new_func()
    b = f.add_block('entry')
    b.add(IRCall(None, 'lp_no_such_function', [], VOID))
    b.add(IRReturn(None))
    expect_error(new_module(f), "unknown callee 'lp_no_such_function'")


def main():
    tests = [v for name, v in sorted(globals().items())
             if name.startswith('test_') and callable(v)]
    for test in tests:
        test()
    print(f"IR verifier: {len(tests)} tests passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
