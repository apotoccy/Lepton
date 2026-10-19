#!/usr/bin/env python3
# =============================================================================
#  leptonc.py -- the Lepton compiler
#
#  Lepton  ->  C11  ->  native code (host gcc/clang)
#
#  Usage:
#      python3 leptonc.py hello.lp -o hello
#      ./hello
#
#      python3 leptonc.py hello.lp -S -o hello.c      # only emit C
#      python3 leptonc.py hello.lp --m32 -o hello32   # 32-bit build
#      python3 leptonc.py hello.lp --freestanding -S  # no host runtime
# =============================================================================

from __future__ import annotations
from lepton_types import PRIMS
from pathlib import Path
import argparse, os, subprocess, sys, tempfile

# When this file is run as a script, __name__ is "__main__" while the
# lowering/emitter passes import `leptonc` by name. Without this alias the
# imported module gets a second CompileError class and errors raised below
# would escape main()'s except clause as tracebacks.
if __name__ == '__main__':
    sys.modules.setdefault('leptonc', sys.modules[__name__])


# =============================================================================
#  Errors
# =============================================================================
class CompileError(Exception):
    def __init__(self, msg, line=0, col=0, fname="<input>"):
        self.msg, self.line, self.col, self.fname = msg, line, col, fname
        super().__init__(f"{fname}:{line}:{col}: error: {msg}")


# =============================================================================
#  Lexer
# =============================================================================
KEYWORDS = {
    'fn', 'let', 'var', 'const', 'if', 'else', 'while', 'for', 'in',
    'return', 'break', 'continue', 'struct', 'enum', 'raw', 'extern',
    'as', 'true', 'false', 'void', 'import', 'alloc',
}

PUNCT = sorted([
    '<<=', '>>=', '..=', '->', '..', '==', '!=', '<=', '>=', '&&', '||',
    '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<', '>>',
    '(', ')', '{', '}', '[', ']', ',', ';', ':', '.', '+', '-', '*', '/',
    '%', '<', '>', '=', '!', '&', '|', '^', '~',
], key=len, reverse=True)

ESCAPES = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"',
           "'": "'", '0': '\0'}


def lex(src: str, fname: str):
    toks = []
    i, line, col, n = 0, 1, 1, len(src)

    def adv(k=1):
        nonlocal i, line, col
        for _ in range(k):
            if i < n and src[i] == '\n':
                line += 1
                col = 1
            else:
                col += 1
            i += 1

    while i < n:
        c = src[i]

        if c in ' \t\r\n':
            adv(); continue

        # comments
        if src.startswith('//', i):
            while i < n and src[i] != '\n':
                adv()
            continue
        if src.startswith('/*', i):
            adv(2)
            while i < n and not src.startswith('*/', i):
                adv()
            adv(2); continue

        # numbers
        if c.isdigit():
            sl, sc, start = line, col, i
            if src.startswith(('0x', '0X'), i):
                adv(2)
                while i < n and (src[i].isdigit() or src[i].lower() in 'abcdef' or src[i] == '_'):
                    adv()
                toks.append(('int', int(src[start:i].replace('_', ''), 16), sl, sc)); continue
            if src.startswith(('0b', '0B'), i):
                adv(2)
                while i < n and src[i] in '01_':
                    adv()
                toks.append(('int', int(src[start + 2:i].replace('_', ''), 2), sl, sc)); continue
            while i < n and (src[i].isdigit() or src[i] == '_'):
                adv()
            isf = False
            if i < n and src[i] == '.' and i + 1 < n and src[i + 1].isdigit():
                isf = True; adv()
                while i < n and (src[i].isdigit() or src[i] == '_'):
                    adv()
            if i < n and src[i] in 'eE':
                isf = True; adv()
                if i < n and src[i] in '+-':
                    adv()
                while i < n and src[i].isdigit():
                    adv()
            txt = src[start:i].replace('_', '')
            # Numeric literal suffixes: `h` for f16, `f` for f32. A
            # suffix must not be followed by an identifier character, so
            # `1hello` still lexes as int(1) then ident(hello).
            if (i < n and src[i] in 'hH'
                    and (i + 1 >= n
                         or not (src[i + 1].isalnum() or src[i + 1] == '_'))):
                adv()
                toks.append(('float16', float(txt), sl, sc))
                continue
            if (i < n and src[i] in 'fF'
                    and (i + 1 >= n
                         or not (src[i + 1].isalnum() or src[i + 1] == '_'))):
                adv()
                toks.append(('float32', float(txt), sl, sc))
                continue
            toks.append(('float' if isf else 'int',
                         float(txt) if isf else int(txt), sl, sc))
            continue
            

        # identifiers / keywords
        if c.isalpha() or c == '_':
            sl, sc, start = line, col, i
            while i < n and (src[i].isalnum() or src[i] == '_'):
                adv()
            w = src[start:i]
            toks.append(('kw' if w in KEYWORDS else 'ident', w, sl, sc))
            continue

        # string literals (with {expr} interpolation)
        if c == '"':
            sl, sc = line, col
            adv()
            parts, buf = [], []
            while True:
                if i >= n:
                    raise CompileError("unterminated string literal", sl, sc, fname)
                ch = src[i]
                if ch == '"':
                    adv(); break
                if ch == '\\':
                    adv()
                    e = src[i]; adv()
                    buf.append(ESCAPES.get(e, e))
                    continue
                if ch == '{':
                    if buf:
                        parts.append(('lit', ''.join(buf))); buf = []
                    adv()
                    depth, sub = 1, []
                    while i < n and depth > 0:
                        if src[i] == '{':
                            depth += 1
                        elif src[i] == '}':
                            depth -= 1
                            if depth == 0:
                                break
                        sub.append(src[i]); adv()
                    if i >= n:
                        raise CompileError("unterminated interpolation", sl, sc, fname)
                    adv()
                    parts.append(('expr', ''.join(sub)))
                    continue
                buf.append(ch); adv()
            if buf:
                parts.append(('lit', ''.join(buf)))
            toks.append(('str', parts, sl, sc))
            continue

        # char literals
        if c == "'":
            sl, sc = line, col
            adv()
            if src[i] == '\\':
                adv(); e = src[i]; adv()
                val = ord(ESCAPES.get(e, e))
            else:
                val = ord(src[i]); adv()
            if i >= n or src[i] != "'":
                raise CompileError("unterminated char literal", sl, sc, fname)
            adv()
            toks.append(('char', val, sl, sc))
            continue

        # punctuation
        for p in PUNCT:
            if src.startswith(p, i):
                toks.append(('op', p, line, col)); adv(len(p)); break
        else:
            raise CompileError(f"unexpected character {c!r}", line, col, fname)

    toks.append(('eof', None, line, col))
    return toks


# =============================================================================
#  AST
# =============================================================================
class Node:
    """AST node. `pos` is a (line, col) tuple, or None for synthetic nodes."""

    def __init__(self, kind, pos=None, **kw):
        self.kind = kind
        self.pos = pos
        self.__dict__.update(kw)

    def __repr__(self):
        return f"<{self.kind}>"


# =============================================================================
#  Parser
# =============================================================================
class Parser:
    BINOPS = [
        ['||'],
        ['&&'],
        ['==', '!='],
        ['<', '<=', '>', '>='],
        ['|'],
        ['^'],
        ['&'],
        ['<<', '>>'],
        ['+', '-'],
        ['*', '/', '%'],
    ]
    ASSIGN_OPS = {'=', '+=', '-=', '*=', '/=', '%=',
                  '&=', '|=', '^=', '<<=', '>>='}

    def __init__(self, toks, fname):
        self.toks, self.pos, self.fname = toks, 0, fname

    # ---- token helpers ----
    def peek(self, k=0):
        return self.toks[min(self.pos + k, len(self.toks) - 1)]

    def at(self, kind, value=None):
        t = self.toks[self.pos]
        return t[0] == kind and (value is None or t[1] == value)

    def at_op(self, v):
        return self.at('op', v)

    def at_kw(self, v):
        return self.at('kw', v)

    def next(self):
        t = self.toks[self.pos]
        self.pos += 1
        return t

    def eat(self, kind, value=None):
        if self.at(kind, value):
            return self.next()
        return None

    def expect(self, kind, value=None):
        t = self.toks[self.pos]
        if t[0] != kind or (value is not None and t[1] != value):
            want = value if value is not None else kind
            raise CompileError(f"expected {want!r} but found {t[1]!r}", t[2], t[3], self.fname)
        self.pos += 1
        return t

    def err(self, msg, t=None):
        t = t or self.peek()
        raise CompileError(msg, t[2], t[3], self.fname)

    # ---- types ----
    def parse_type(self):
        if self.at_op('['):
            self.next()
            elem = self.parse_type()
            self.expect('op', ';')
            cnt = self.expect('int')[1]
            self.expect('op', ']')
            return ('array', elem, cnt)

        t = self.peek()
        if t[0] == 'kw' and t[1] == 'void':
            self.next()
            return ('prim', 'void')
        if t[0] != 'ident':
            self.err(f"expected a type, found {t[1]!r}")
        self.next()
        name = t[1]
        if name == 'ptr':
            self.expect('op', '[')
            inner = self.parse_type()
            self.expect('op', ']')
            return ('ptr', inner)
        if name in PRIMS:
            return ('prim', name)
        return ('named', name)

    # ---- top level ----
    def parse_program(self):
        items = []
        while not self.at('eof'):
            items.append(self.parse_item())
        return items
        
    def parse_params(self):
        self.expect('op', '(')
        params = []
        if not self.at_op(')'):
            while True:
                nm_tok = self.expect('ident')
                nm = nm_tok[1]
                self.expect('op', ':')
                ty = self.parse_type()
                params.append(Node('param', pos=(nm_tok[2], nm_tok[3]),
                                   name=nm, type=ty))
                if not self.eat('op', ','):
                    break
                if self.at_op(')'):
                    break
        self.expect('op', ')')
        return params
        
        
    def parse_item(self):
        t = self.peek()
        start = (t[2], t[3])

        if t[0] == 'kw' and t[1] == 'import':
            self.next()
            if self.at('str'):
                parts = self.next()[1]
                if any(k == 'expr' for k, _ in parts):
                    self.err("import path cannot contain interpolation")
                path = ''.join(v for k, v in parts if k == 'lit')
            else:
                segs = [self.expect('ident')[1]]
                while self.eat('op', '.'):
                    segs.append(self.expect('ident')[1])
                path = '/'.join(segs) + '.lp'
            self.expect('op', ';')
            return Node('import', pos=start, path=path)

        if t[0] == 'kw' and t[1] == 'extern':
            self.next()
            self.expect('kw', 'fn')
            name = self.expect('ident')[1]
            params = self.parse_params()
            ret = None
            if self.eat('op', '->'):
                ret = self.parse_type()
            self.expect('op', ';')
            return Node('extern', pos=start, name=name, params=params, ret=ret)

        if t[0] == 'kw' and t[1] == 'fn':
            self.next()
            name = self.expect('ident')[1]
            params = self.parse_params()
            ret = None
            if self.eat('op', '->'):
                ret = self.parse_type()
            body = self.parse_block()
            return Node('fn', pos=start, name=name, params=params,
                        ret=ret, body=body)

        if t[0] == 'kw' and t[1] == 'struct':
            self.next()
            name = self.expect('ident')[1]
            self.expect('op', '{')
            fields = []
            while not self.at_op('}'):
                fn = self.expect('ident')[1]
                self.expect('op', ':')
                ft = self.parse_type()
                self.expect('op', ';')
                fields.append((fn, ft))
            self.expect('op', '}')
            return Node('struct', pos=start, name=name, fields=fields)

        if t[0] == 'kw' and t[1] == 'enum':
            self.next()
            name = self.expect('ident')[1]
            self.expect('op', '{')
            variants = []
            while not self.at_op('}'):
                variants.append(self.expect('ident')[1])
                if not self.eat('op', ','):
                    break
                if self.at_op('}'):
                    break
            self.expect('op', '}')
            return Node('enum', pos=start, name=name, variants=variants)

        if t[0] == 'kw' and t[1] == 'const':
            self.next()
            name = self.expect('ident')[1]
            self.expect('op', ':')
            ty = self.parse_type()
            self.expect('op', '=')
            val = self.parse_expr()
            self.expect('op', ';')
            return Node('const', pos=start, name=name, type=ty, value=val)

        self.err(f"expected a top-level declaration, found {t[1]!r}")

    # ---- statements ----
    
    def parse_block(self, pos=None):
        if pos is None:
            t = self.peek()
            pos = (t[2], t[3])
        self.expect('op', '{')
        stmts = []
        while not self.at_op('}'):
            stmts.append(self.parse_stmt())
        self.expect('op', '}')
        return Node('block', pos=pos, stmts=stmts)
    

    def parse_stmt(self):
        t = self.peek()
        start = (t[2], t[3])

        if t[0] == 'kw':
            w = t[1]

            if w in ('let', 'var'):
                self.next()
                name = self.expect('ident')[1]
                ty = None
                if self.eat('op', ':'):
                    ty = self.parse_type()
                val = None
                if self.eat('op', '='):
                    val = self.parse_expr()
                self.expect('op', ';')
                if ty is None and val is None:
                    self.err("'let' needs either a type or an initialiser")
                return Node('let', pos=start, name=name, type=ty, value=val,
                            mutable=(w == 'var'))

            if w == 'if':
                self.next()
                self.expect('op', '(')
                cond = self.parse_expr()
                self.expect('op', ')')
                then = self.parse_block()
                els = None
                if self.at_kw('else'):
                    self.next()
                    els = (self.parse_stmt() if self.at_kw('if')
                           else self.parse_block())
                return Node('if', pos=start, cond=cond, then=then, els=els)

            if w == 'while':
                self.next()
                self.expect('op', '(')
                cond = self.parse_expr()
                self.expect('op', ')')
                body = self.parse_block()
                return Node('while', pos=start, cond=cond, body=body)

            if w == 'for':
                self.next()
                var = self.expect('ident')[1]
                self.expect('kw', 'in')
                lo = self.parse_expr()
                inclusive = False
                if self.eat('op', '..'):
                    pass
                elif self.eat('op', '..='):
                    inclusive = True
                else:
                    self.err("expected '..' or '..=' in for-loop")
                hi = self.parse_expr()
                body = self.parse_block()
                return Node('for', pos=start, var=var, lo=lo, hi=hi,
                            inclusive=inclusive, body=body)

            if w == 'return':
                self.next()
                val = None if self.at_op(';') else self.parse_expr()
                self.expect('op', ';')
                return Node('return', pos=start, value=val)

            if w == 'break':
                self.next(); self.expect('op', ';')
                return Node('break', pos=start)

            if w == 'continue':
                self.next(); self.expect('op', ';')
                return Node('continue', pos=start)

            if w == 'raw':
                self.next()
                return Node('raw', pos=start, body=self.parse_block())

        if t[0] == 'op' and t[1] == '{':
            return self.parse_block(pos=start)

	# expression / assignment statement
        e = self.parse_expr()
        if self.at('op') and self.peek()[1] in self.ASSIGN_OPS:
            op = self.next()[1]
            val = self.parse_expr()
            self.expect('op', ';')
            return Node('assign', pos=start, target=e, op=op, value=val)
        self.expect('op', ';')
        return Node('exprstmt', pos=start, expr=e)

    # ---- expressions ----
    def parse_expr(self):
        return self.parse_bin(0)

    def parse_expr_full(self):
        e = self.parse_expr()
        if not self.at('eof'):
            self.err("trailing tokens in interpolated expression")
        return e
        
    def parse_bin(self, lvl):
        if lvl >= len(self.BINOPS):
            return self.parse_unary()
        left = self.parse_bin(lvl + 1)
        while self.at('op') and self.peek()[1] in self.BINOPS[lvl]:
            op_tok = self.next()
            op = op_tok[1]
            right = self.parse_bin(lvl + 1)
            left = Node('bin', pos=(op_tok[2], op_tok[3]),
                        op=op, left=left, right=right)
        return left
        
    def parse_unary(self):
        if self.at('op') and self.peek()[1] in ('-', '!', '~', '&', '*'):
            op_tok = self.next()
            return Node('un', pos=(op_tok[2], op_tok[3]),
                        op=op_tok[1], expr=self.parse_unary())
        return self.parse_postfix()
        
    def parse_postfix(self):
        e = self.parse_primary()
        while True:
            if self.at_op('('):
                self.next()
                args = []
                if not self.at_op(')'):
                    while True:
                        if (self.at('ident') and self.peek(1)[0] == 'op'
                                and self.peek(1)[1] == ':'):
                            nm = self.next()[1]
                            self.next()
                            args.append(Node('namedarg',
                                             pos=e.pos,
                                             name=nm,
                                             value=self.parse_expr()))
                        else:
                            args.append(self.parse_expr())
                        if not self.eat('op', ','):
                            break
                        if self.at_op(')'):
                            break
                self.expect('op', ')')
                e = Node('call', pos=e.pos, callee=e, args=args)

            elif self.at_op('['):
                self.next()
                idx = self.parse_expr()
                self.expect('op', ']')
                e = Node('index', pos=e.pos, base=e, index=idx)

            elif self.at_op('.'):
                self.next()
                nm = self.expect('ident')[1]
                e = Node('field', pos=e.pos, base=e, name=nm)

            elif self.at_kw('as'):
                self.next()
                ty = self.parse_type()
                e = Node('cast', pos=e.pos, expr=e, type=ty)

            else:
                break
        return e
    
    

    def parse_primary(self):
        t = self.peek()
        start = (t[2], t[3])

        if t[0] == 'int':
            self.next(); return Node('int', pos=start, value=t[1])
        if t[0] == 'float':
            self.next(); return Node('float', pos=start, value=t[1])
        if t[0] == 'float32':
            self.next(); return Node('float32', pos=start, value=t[1])
        if t[0] == 'float16':
            self.next(); return Node('float16', pos=start, value=t[1])
        if t[0] == 'char':
            self.next(); return Node('charlit', pos=start, value=t[1])
        if t[0] == 'bool':
            self.next(); return Node('bool', pos=start, value=t[1])

        if t[0] == 'kw' and t[1] in ('true', 'false'):
            self.next()
            return Node('bool', pos=start, value=(t[1] == 'true'))

        if t[0] == 'str':
            self.next()
            parts = []
            for kind, v in t[1]:
                if kind == 'lit':
                    parts.append(('lit', v))
                else:
                    sub = Parser(lex(v, self.fname), self.fname).parse_expr_full()
                    parts.append(('expr', sub))
            return Node('strlit', pos=start, parts=parts)

        if t[0] == 'ident':
            self.next()
            return Node('ident', pos=start, name=t[1])

        if self.at_op('('):
            self.next()
            e = self.parse_expr()
            self.expect('op', ')')
            return e

        if t[0] == 'kw' and t[1] == 'alloc':
            self.next()
            self.expect('op', '(')
            ty = self.parse_type()
            self.expect('op', ')')
            return Node('alloc', pos=start, type=ty)
            
        if self.at_op('['):
            self.next()
            elems = []
            if not self.at_op(']'):
                while True:
                    elems.append(self.parse_expr())
                    if not self.eat('op', ','):
                        break
                    if self.at_op(']'):
                        break
            self.expect('op', ']')
            if not elems:
                self.err("empty vector literal", start)
            return Node('veclit', pos=start, elems=elems)
            
        if self.at_op('{'):
            self.next()
            elems = []
            if not self.at_op('}'):
                while True:
                    elems.append(self.parse_expr())
                    if not self.eat('op', ','):
                        break
                    if self.at_op('}'):
                        break
            self.expect('op', '}')
            if not elems:
                self.err("empty array literal", start)
            return Node('arrlit', pos=start, elems=elems)            

        self.err(f"unexpected token {t[1]!r}")


# =============================================================================
#  Driver
# =============================================================================
class ModuleLoader:
    """Resolves and loads imported modules once, in depth-first order."""

    def __init__(self, search_paths):
        self.search_paths = [Path(p).resolve() for p in search_paths]
        self.visited = set()
        self.loading = set()

    def find(self, rel_path, base_dir):
        candidate = (base_dir / rel_path).resolve()
        if candidate.exists():
            return candidate
        for sp in self.search_paths:
            candidate = (sp / rel_path).resolve()
            if candidate.exists():
                return candidate
        raise CompileError(f"cannot find module '{rel_path}'")

    def load(self, path, out):
        path = Path(path).resolve()
        if path in self.visited:
            return
        if path in self.loading:
            raise CompileError(f"circular import: {path}")
        self.loading.add(path)
        try:
            src = path.read_text()
            toks = lex(src, str(path))
            items = Parser(toks, str(path)).parse_program()
            for it in items:
                if it.kind == 'import':
                    dep = self.find(it.path, path.parent)
                    self.load(dep, out)
                else:
                    out.append(it)
            self.visited.add(path)
        finally:
            self.loading.discard(path)


def default_search_paths():
    paths = []
    env = os.environ.get('LEPTON_PATH', '')
    if env:
        paths.extend(env.split(os.pathsep))
    paths.append('.')
    paths.append('./std')
    return paths


def flatten_imports(items, fname, search_paths):
    """Resolve `import` nodes depth-first, returning a flat item list."""
    loader = ModuleLoader(search_paths)
    loader.visited.add(Path(fname).resolve())
    flat = []
    base_dir = Path(fname).resolve().parent
    for it in items:
        if it.kind == 'import':
            dep = loader.find(it.path, base_dir)
            loader.load(dep, flat)
        else:
            flat.append(it)
    return flat

def compile_to_ir(src, fname, search_paths=None):
    """Compile source text to an IR module. Useful for --dump-ir and
    other tooling that wants to inspect or transform the IR."""
    toks = lex(src, fname)
    items = Parser(toks, fname).parse_program()
    if search_paths is not None:
        items = flatten_imports(items, fname, search_paths)
    from lepton_lower import lower_module
    return lower_module(items, fname)

def compile_source(src, fname, freestanding=False, search_paths=None,
                   target=None):
    """Front-to-back compilation: source text -> C source text.

    The pipeline is lex -> parse -> flatten imports -> lower to IR ->
    emit C. The IR modules are imported lazily to avoid an import cycle:
    lepton_lower and lepton_c_ir both import CompileError from this file.
    """
    from lepton_c_ir import compile_ir
    module = compile_to_ir(src, fname, search_paths)
    return compile_ir(module, fname, freestanding, target)
# deprecated
#     toks = lex(src, fname)
#     items = Parser(toks, fname).parse_program()
#     if search_paths is not None:
#         items = flatten_imports(items, fname, search_paths)
#
#     from lepton_lower import lower_module
#     from lepton_c_ir import compile_ir
#
#     module = lower_module(items, fname)
#     return compile_ir(module, fname, freestanding, target)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='leptonc', description='Compile Lepton source code.')
    ap.add_argument('input', help='.lp source file')
    ap.add_argument('-o', '--output', help='output file')
    ap.add_argument('-S', '--emit-c', action='store_true',
                    help='emit C source instead of compiling')
    ap.add_argument('--cc', default=os.environ.get('CC', 'cc'),
                    help='C compiler to use (default: $CC or cc)')
    ap.add_argument('-O', '--opt', default='2', help='optimisation level')
    ap.add_argument('--m32', action='store_true', help='build a 32-bit binary')
    ap.add_argument('--freestanding', action='store_true',
                    help='do not emit the host runtime (for kernels)')
    ap.add_argument('--target', default='scalar',
                    choices=['scalar', 'x86_64'],
                    help='vector lowering target (default: scalar)')
    ap.add_argument('--dump-ir', action='store_true',
                    help='print the lowered IR instead of compiling')
    a = ap.parse_args(argv)

    try:
        with open(a.input, 'r') as f:
            src = f.read()
    except OSError as e:
        print(f"leptonc: cannot read {a.input}: {e}", file=sys.stderr)
        return 1

    try:
        if a.dump_ir:
            from lepton_ir import render_module
            module = compile_to_ir(src, a.input,
                                   search_paths=default_search_paths())
            text = render_module(module)
            if a.output:
                with open(a.output, 'w') as f:
                    f.write(text)
                print(f"wrote {a.output}")
            else:
                sys.stdout.write(text)
            return 0
        csrc = compile_source(src, a.input, a.freestanding,
                              search_paths=default_search_paths(),
                              target=a.target)
    except CompileError as e:
        print(str(e), file=sys.stderr)
        return 1

    if a.emit_c:
        if a.output:
            with open(a.output, 'w') as f:
                f.write(csrc)
            print(f"wrote {a.output}")
        else:
            sys.stdout.write(csrc)
        return 0

    out = a.output or os.path.splitext(os.path.basename(a.input))[0]

    with tempfile.TemporaryDirectory() as td:
        cf = os.path.join(td, 'lepton_out.c')
        with open(cf, 'w') as f:
            f.write(csrc)

        cmd = [a.cc, '-std=c11']
        if a.m32:
            cmd.append('-m32')
        cmd += [f'-O{a.opt}', '-o', out, cf, '-lm']

        r = subprocess.run(cmd)
        if r.returncode != 0:
            print("leptonc: C backend failed", file=sys.stderr)
            return r.returncode

    print(f"built {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
