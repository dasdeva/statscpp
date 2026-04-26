"""
Compiler detection and compilation.

Supports gcc-compatible compilers (g++, clang++) on all platforms and MSVC
(cl.exe) on Windows as a fallback.
"""
import platform
import shutil
import subprocess
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"
IS_MACOS   = platform.system() == "Darwin"


class CompilationError(RuntimeError):
    """Raised when the C++ compiler rejects user code."""


def find() -> tuple[str, str]:
    """Locate a C++ compiler and return (path, kind).

    kind is ``'gcc'`` for g++/clang++ or ``'msvc'`` for cl.exe.
    Raises RuntimeError with install instructions if none is found.
    """
    for name in ("g++", "clang++"):
        path = shutil.which(name)
        if path:
            return path, "gcc"
    if IS_WINDOWS:
        path = shutil.which("cl")
        if path:
            return path, "msvc"
    raise RuntimeError(
        "❌ No C++ compiler found on PATH.\n\n"
        "statscpp needs a C++ compiler to turn your code into machine code.\n"
        "Install one for your OS:\n\n"
        "  macOS:\n"
        "    xcode-select --install\n\n"
        "  Ubuntu/Debian:\n"
        "    sudo apt install build-essential\n\n"
        "  Fedora/RHEL:\n"
        "    sudo dnf install gcc-c++\n\n"
        "  Arch:\n"
        "    sudo pacman -S base-devel\n\n"
        "  Windows:\n"
        "    Option 1: Install MinGW-w64\n"
        "      winget install MSYS2.MSYS2\n\n"
        "    Option 2: Install Visual Studio Build Tools\n"
        "      (includes cl.exe compiler)"
    )


def flags(kind: str) -> list[str]:
    """Return compiler flags for building a shared library."""
    if kind == "msvc":
        return ["/O2", "/EHsc", "/std:c++17", "/LD", "/nologo"]
    result = ["-O2", "-std=c++17", "-shared", "-fPIC"]
    if IS_MACOS:
        result += ["-undefined", "dynamic_lookup"]
    return result


def suffix() -> str:
    """Return the platform-appropriate shared-library file extension."""
    if IS_WINDOWS: return ".dll"
    if IS_MACOS:   return ".dylib"
    return ".so"


def compile(
    src: str,
    lib_path: Path,
    src_path: Path,
    extra_flags: list[str] | None = None,
) -> None:
    """Write *src* to *src_path*, compile to *lib_path*.

    Raises CompilationError on non-zero exit.
    """
    compiler_path, kind = find()
    src_path.write_text(src, encoding="utf-8")
    extra = extra_flags or []

    if kind == "msvc":
        cmd = [compiler_path] + flags(kind) + extra + [str(src_path), f"/Fe:{lib_path}"]
    else:
        cmd = [compiler_path] + flags(kind) + extra + [str(src_path), "-o", str(lib_path)]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(lib_path.parent))
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise CompilationError(
            f"❌ C++ compiler error. Your code has a syntax or type error:\n\n{stderr}\n\n"
            f"Generated C++ source saved at: {src_path}\n"
            f"You can inspect this file to debug the issue."
        )
