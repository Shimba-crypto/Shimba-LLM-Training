"""
compat.py — Console encoding shim.

Windows consoles still default to cp1252, so any non-ASCII character in a
progress banner (the "✓" in a checkpoint message, an arrow, an em-dash)
raises UnicodeEncodeError and kills an otherwise-fine training run.

`enable_safe_output()` reconfigures stdout/stderr to UTF-8 with a replace
fallback. On Linux/Colab this is a no-op; on Windows it turns a crash into
harmless '?' characters.
"""

from __future__ import annotations

import sys


def enable_safe_output() -> None:
    """Make stdout/stderr tolerant of non-ASCII. Safe to call repeatedly."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Stream is detached or already closed — nothing useful to do.
            pass