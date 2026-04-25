"""
Armadillo detection.

Locates the Armadillo C++ linear-algebra library via pkg-config (preferred)
or by probing common install paths.  The result is cached after the first
successful lookup so repeated calls are cheap.
"""
import shutil
import subprocess
from pathlib import Path

from . import types

# Module-level cache: None means "not yet searched"
_cache: tuple[list[str], list[str]] | None = None

# Common (include_dir, lib_dir) candidates in priority order
_CANDIDATES = [
    ("/opt/homebrew/include", "/opt/homebrew/lib"),   # macOS Apple-Silicon brew
    ("/usr/local/include",    "/usr/local/lib"),       # macOS Intel brew / manual
    ("/usr/include",          "/usr/lib"),             # Linux system packages
    ("/usr/include",          "/usr/lib/x86_64-linux-gnu"),  # Debian/Ubuntu multiarch
]

_INSTALL_HINT = (
    "Armadillo not found. Install it first:\n"
    "  macOS:  brew install armadillo\n"
    "  Ubuntu: sudo apt install libarmadillo-dev\n"
    "  Fedora: sudo dnf install armadillo-devel\n"
    "  Arch:   sudo pacman -S armadillo\n"
    "Then restart Python (or clear statscpp's module-level cache)."
)


def find() -> tuple[list[str], list[str]]:
    """Return (cflags, ldflags) needed to compile against Armadillo.

    Result is cached after the first successful call.
    Raises RuntimeError with install instructions if Armadillo is not found.
    """
    global _cache
    if _cache is not None:
        return _cache

    # --- pkg-config (most reliable when available) ---
    pkg = shutil.which("pkg-config")
    if pkg:
        r = subprocess.run(
            [pkg, "--cflags", "--libs", "armadillo"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            parts   = r.stdout.split()
            cflags  = [p for p in parts if p.startswith("-I")]
            ldflags = [p for p in parts if not p.startswith("-I")]
            _cache  = (cflags, ldflags)
            return _cache

    # --- Probe common paths ---
    for inc, lib in _CANDIDATES:
        if (Path(inc) / "armadillo").exists():
            _cache = ([f"-I{inc}"], [f"-L{lib}", "-larmadillo"])
            return _cache

    raise RuntimeError(_INSTALL_HINT)


def needed(code: str, funcs: list) -> bool:
    """Return True if *code* or any function signature requires Armadillo."""
    if "arma::" in code:
        return True
    return any(
        tag in (types.AVEC, types.AMAT)
        for _, _, params in funcs
        for tag, _ in params
    )
