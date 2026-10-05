"""
vida_compute.py — Python bridge to the vida (.vda) bytecode VM.

vida lives in vendor/vida (git submodule) and beats CPython ~3x on tight
numeric loops. This module does NOT accelerate torch itself — matmuls stay
where they belong. It covers the numeric work around training instead:

  * independently re-deriving arithmetic answers (a second engine besides
    Python, so a dataset bug has to fool two runtimes to survive),
  * scoring eval outputs (exact-match over thousands of rows without
    paying CPython loop overhead per row).

When no `vda` binary is present (fresh clone without building vendor/vida,
Colab without the submodule), every helper falls back to pure Python and
keeps working — slower, never broken. Build it with `make -C vendor/vida`
(or `wsl make -C vendor/vida` on Windows).
"""

from __future__ import annotations

import os
import shutil
import subprocess

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set VDA_BIN to point flash-slms at any vida binary explicitly:
#   VDA_BIN=/path/to/vda  (or vda.exe on Windows)
# Otherwise the repo's vendor/vida build is used, then PATH.
ENV_VAR = "VDA_BIN"


def _vendor_candidates():
    exe = "vda.exe" if os.name == "nt" else "vda"
    yield os.path.join(_REPO_ROOT, "vendor", "vida", "build", exe)


def find_vda() -> str | None:
    """Path to a runnable `vda` binary, or None (callers fall back)."""
    explicit = os.environ.get(ENV_VAR)
    if explicit and os.path.isfile(explicit):
        return explicit
    for cand in _vendor_candidates():
        if cand and os.path.isfile(cand):
            return cand
    if os.name == "nt":
        return shutil.which("vda.exe") or shutil.which("vda")
    return shutil.which("vda")


def vda_available() -> bool:
    return find_vda() is not None


def vda_eval(expr: str, timeout: int = 10) -> str:
    """
    Evaluate one vida expression, return stdout stripped.

    Raises RuntimeError when no binary exists or the script fails, so
    callers can fall back to Python. `expr` should be a pure expression
    (e.g. "7*8", "sqrt(3*3+4*4)"); it is wrapped in print() here.
    """
    exe = find_vda()
    if exe is None:
        raise RuntimeError("no vda binary (build vendor/vida first)")
    proc = subprocess.run(
        [exe, "eval", f"print({expr})"],
        capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"vda failed on {expr!r}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _numbers_equal(a: str, b: str) -> bool:
    """Numeric comparison tolerant of 56 vs 56.0 style formatting."""
    try:
        return float(a) == float(b)
    except (ValueError, TypeError):
        return a.strip() == b.strip()


def verify_rows(pairs: list, use_vida: bool = True) -> dict:
    """
    Re-derive expected answers through vida (or Python as fallback).

    pairs: [(vda_expression, expected_string), ...].
    Returns {"hits": int, "total": int, "engine": "vda"|"python",
             "mismatches": [(expr, expected, got), ...]}.
    A mismatch means the DATASET row disagrees with an independent engine —
    fix the row, not the model.
    """
    engine, hits, mismatches = "python", 0, []
    vda_ok = use_vida and vda_available()
    if vda_ok:
        # Pre-flight: a present-but-unrunnable binary must not poison every
        # row into a false mismatch. One probe decides the whole run.
        try:
            assert vda_eval("1+1") == "2"
            engine = "vda"
        except Exception:
            vda_ok = False
    import math as _math
    _ns = {k: getattr(_math, k) for k in dir(_math) if not k.startswith("_")}
    for expr, expected in pairs:
        try:
            got = (vda_eval(expr) if vda_ok
                   else str(eval(expr, {"__builtins__": {}}, dict(_ns))))
        except Exception:
            got = None
        if got is not None and _numbers_equal(got, expected):
            hits += 1
        else:
            mismatches.append((expr, expected, got))
    return {"hits": hits, "total": len(pairs), "engine": engine,
            "mismatches": mismatches}
