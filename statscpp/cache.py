"""
Compile-once cache.

Compiled shared libraries are stored in ~/.cache/statscpp/ keyed by the
MD5 hash of (package_version + source_code).  This makes the cache
version-aware: upgrading statscpp automatically invalidates old .so files.
"""
import ctypes
import hashlib
from pathlib import Path

from . import compiler as _compiler
from ._version import __version__

CACHE_DIR: Path = Path.home() / ".cache" / "statscpp"

# In-process library cache: key → loaded ctypes.CDLL
_libs: dict[str, ctypes.CDLL] = {}


def make_key(code: str) -> str:
    """Return a stable cache key for *code* scoped to the current version."""
    return hashlib.md5((__version__ + code.strip()).encode()).hexdigest()


def get(key: str) -> ctypes.CDLL | None:
    """Return the cached library for *key*, or None if not yet compiled."""
    return _libs.get(key)


def load(
    key: str,
    src: str,
    extra_flags: list[str] | None = None,
) -> ctypes.CDLL:
    """Return the compiled library for *key*, compiling from *src* if needed.

    The result is kept in both the on-disk cache (CACHE_DIR) and the
    in-process dict so repeated calls within the same session are instant.
    """
    if key in _libs:
        return _libs[key]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lib_path = CACHE_DIR / f"{key}{_compiler.suffix()}"
    src_path = CACHE_DIR / f"{key}.cpp"

    if not lib_path.exists():
        _compiler.compile(src, lib_path, src_path, extra_flags)

    lib = ctypes.CDLL(str(lib_path))
    _libs[key] = lib
    return lib
