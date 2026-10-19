#!/usr/bin/env python3
"""
Lepton test harness.

Layout:
    tests/cases/    *.lp programs. A matching .out file gives expected stdout.
                    Programs without a .out file are skipped (use these as
                    shared library modules that other cases import).
    tests/errors/   *.lp programs that must fail to compile. A matching .err
                    file gives a substring that must appear in the error.

Usage:
    python3 tests/run_tests.py
    python3 tests/run_tests.py --filter arith
    python3 tests/run_tests.py -v
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
LEPTONC = ROOT / "leptonc.py"
PYTHON  = sys.executable
TARGET  = os.environ.get("LEPTON_TARGET", "scalar")


def run_case(lp_file, out_file):
    """Compile and run a program, diff stdout against expected."""
    with tempfile.TemporaryDirectory() as td:
        binary = os.path.join(td, "prog")
        r = subprocess.run(
            [PYTHON, str(LEPTONC), str(lp_file), "-o", binary,
             "--target", TARGET],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return False, f"compile failed:\n{r.stderr}"
        r = subprocess.run([binary], capture_output=True, text=True,
                           timeout=10, cwd=td)
        exit_file = lp_file.with_suffix(".exit")
        expected_exit = 0
        if exit_file.exists():
            expected_exit = int(exit_file.read_text().strip())
        if r.returncode != expected_exit:
            return False, (f"runtime exit {r.returncode}, "
                           f"expected {expected_exit}:\n{r.stderr}")
        expected = out_file.read_text()
        if r.stdout != expected:
            return False, (
                f"stdout mismatch\n"
                f"--- expected ---\n{expected!r}\n"
                f"--- actual ---\n{r.stdout!r}"
            )
        return True, ""


def run_error_case(lp_file, err_file):
    """Compile a program, expect failure, check error message substring."""
    r = subprocess.run(
        [PYTHON, str(LEPTONC), str(lp_file), "-S", "--target", TARGET],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode == 0:
        return False, "expected compile error, got success"
    expected = err_file.read_text().strip()
    if expected not in r.stderr:
        return False, (
            f"error message mismatch\n"
            f"--- expected substring ---\n{expected}\n"
            f"--- actual stderr ---\n{r.stderr}"
        )
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    if not LEPTONC.exists():
        print(f"error: {LEPTONC} not found", file=sys.stderr)
        return 1

    cases_dir  = ROOT / "tests" / "cases"
    errors_dir = ROOT / "tests" / "errors"

    passed, failed, failures = 0, 0, []

    if cases_dir.is_dir():
        for lp in sorted(cases_dir.glob("*.lp")):
            if a.filter and a.filter not in lp.stem:
                continue
            out = lp.with_suffix(".out")
            if not out.exists():
                if a.verbose:
                    print(f"SKIP cases/{lp.name} (no .out)")
                continue
            ok, msg = run_case(lp, out)
            name = f"cases/{lp.name}"
            if ok:
                passed += 1
                if a.verbose:
                    print(f"PASS {name}")
            else:
                failed += 1
                failures.append((name, msg))
                print(f"FAIL {name}")

    if errors_dir.is_dir():
        for lp in sorted(errors_dir.glob("*.lp")):
            if a.filter and a.filter not in lp.stem:
                continue
            err = lp.with_suffix(".err")
            if not err.exists():
                if a.verbose:
                    print(f"SKIP errors/{lp.name} (no .err)")
                continue
            ok, msg = run_error_case(lp, err)
            name = f"errors/{lp.name}"
            if ok:
                passed += 1
                if a.verbose:
                    print(f"PASS {name}")
            else:
                failed += 1
                failures.append((name, msg))
                print(f"FAIL {name}")

    print(f"\n{passed} passed, {failed} failed")
    for name, msg in failures:
        print(f"\n=== {name} ===\n{msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
