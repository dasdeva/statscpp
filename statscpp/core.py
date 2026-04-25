import ctypes
import hashlib
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CACHE_DIR = Path.home() / ".cache" / "statscpp"
_MAX_RESULT = 1_000_000  # pre-allocated output buffer (8 MB of doubles)

_lib_cache: dict[str, ctypes.CDLL] = {}

try:
    from importlib.metadata import version as _pkg_version
    _VERSION = _pkg_version("statscpp")
except Exception:
    _VERSION = "dev"

_IS_WINDOWS = platform.system() == "Windows"
_IS_MACOS   = platform.system() == "Darwin"

# Standard headers always available to user code
_DEFAULT_INCLUDES = """\
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
"""

# DLL-export macro: required by MSVC, no-op elsewhere
_EXPORT_MACRO = """\
#ifdef _WIN32
#  define SCPP_EXPORT __declspec(dllexport)
#else
#  define SCPP_EXPORT
#endif
"""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class CompilationError(RuntimeError):
    """Raised when the C++ compiler rejects user code."""


# ---------------------------------------------------------------------------
# Compiler detection
# ---------------------------------------------------------------------------
def _find_compiler() -> tuple[str, str]:
    """Return (compiler_path, compiler_kind) where kind is 'gcc' or 'msvc'."""
    # Prefer gcc-compatible first (g++, clang++) — works on Linux/macOS/MinGW
    for name in ("g++", "clang++"):
        path = shutil.which(name)
        if path:
            return path, "gcc"
    # Fall back to MSVC cl.exe on Windows
    if _IS_WINDOWS:
        path = shutil.which("cl")
        if path:
            return path, "msvc"
    raise RuntimeError(
        "No C++ compiler found on PATH.\n"
        "  macOS:   xcode-select --install\n"
        "  Ubuntu:  sudo apt install build-essential\n"
        "  Fedora:  sudo dnf install gcc-c++\n"
        "  Windows: install MinGW-w64 (via winget install MSYS2.MSYS2)\n"
        "           or Visual Studio Build Tools, then re-open your terminal."
    )


def _compile_flags(kind: str) -> list[str]:
    if kind == "msvc":
        # /LD = build DLL, /EHsc = C++ exceptions, /nologo = quiet
        return ["/O2", "/EHsc", "/std:c++17", "/LD", "/nologo"]
    flags = ["-O2", "-std=c++17", "-shared", "-fPIC"]
    if _IS_MACOS:
        flags += ["-undefined", "dynamic_lookup"]
    return flags


def _lib_suffix(kind: str) -> str:
    if _IS_WINDOWS:
        return ".dll"
    if _IS_MACOS:
        return ".dylib"
    return ".so"


def _compile(src: str, lib_path: Path, src_path: Path,
             extra_flags: list[str] | None = None) -> None:
    compiler, kind = _find_compiler()
    src_path.write_text(src, encoding="utf-8")
    extra = extra_flags or []

    if kind == "msvc":
        cmd = [compiler] + _compile_flags(kind) + extra + [str(src_path), f"/Fe:{lib_path}"]
    else:
        cmd = [compiler] + _compile_flags(kind) + extra + [str(src_path), "-o", str(lib_path)]

    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_CACHE_DIR))
    if r.returncode != 0:
        stderr = r.stderr.strip() or r.stdout.strip()
        raise CompilationError(f"Compilation failed:\n\n{stderr}")


# ---------------------------------------------------------------------------
# Armadillo detection (result cached after first call)
# ---------------------------------------------------------------------------
_arma_flags: 'tuple[list[str], list[str]] | None' = None  # (cflags, ldflags)


def _find_armadillo() -> tuple[list[str], list[str]]:
    global _arma_flags
    if _arma_flags is not None:
        return _arma_flags

    # Try pkg-config first (works on Linux, macOS with brew's pkg-config)
    pkg = shutil.which("pkg-config")
    if pkg:
        r = subprocess.run(
            [pkg, "--cflags", "--libs", "armadillo"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            parts = r.stdout.split()
            cflags  = [p for p in parts if p.startswith("-I")]
            ldflags = [p for p in parts if not p.startswith("-I")]
            _arma_flags = (cflags, ldflags)
            return _arma_flags

    # Fall back to common install paths
    candidates = [
        ("/opt/homebrew/include", "/opt/homebrew/lib"),   # macOS Apple Silicon brew
        ("/usr/local/include",    "/usr/local/lib"),       # macOS Intel brew / manual
        ("/usr/include",          "/usr/lib"),             # Linux system
        ("/usr/include/armadillo_bits/..", "/usr/lib"),    # some distros
    ]
    for inc, lib in candidates:
        if (Path(inc) / "armadillo").exists():
            _arma_flags = ([f"-I{inc}"], [f"-L{lib}", "-larmadillo"])
            return _arma_flags

    raise RuntimeError(
        "Armadillo not found. Install it first:\n"
        "  macOS:  brew install armadillo\n"
        "  Ubuntu: sudo apt install libarmadillo-dev\n"
        "  Fedora: sudo dnf install armadillo-devel\n"
        "Then restart Python."
    )


def _needs_armadillo(code: str, funcs: list) -> bool:
    if "arma::" in code:
        return True
    return any(
        ptype in ("avec", "amat")
        for _, _, params in funcs
        for ptype, _ in params
    )


# ---------------------------------------------------------------------------
# Type system
# ---------------------------------------------------------------------------
_TYPE_PAT = (
    r'(?:'
    r'(?:std::)?vector\s*<\s*double\s*>'
    r'|(?:std::)?vector\s*<\s*int\s*>'
    r'|arma::mat'
    r'|arma::vec'
    r'|double|float|int|long|size_t'
    r')'
)


def _norm_type(t: str) -> str:
    t = re.sub(r'\s+', ' ', t.strip())
    if re.search(r'vector\s*<\s*double\s*>', t):
        return 'vec'
    if re.search(r'vector\s*<\s*int\s*>', t):
        return 'ivec'
    if t == 'arma::vec':
        return 'avec'
    if t == 'arma::mat':
        return 'amat'
    if t in ('double', 'float'):
        return 'double'
    if t in ('int', 'long', 'size_t'):
        return 'int'
    raise ValueError(
        f"Unsupported type {t!r}. "
        "Supported: int, double, std::vector<double>, std::vector<int>, "
        "arma::vec, arma::mat."
    )


# ---------------------------------------------------------------------------
# Signature parsing
# ---------------------------------------------------------------------------
_FUNC_RE = re.compile(
    rf'({_TYPE_PAT})\s+(\w+)\s*\(([^{{;]*?)\)\s*(?:const\s*)?{{',
    re.MULTILINE | re.DOTALL,
)


def _parse_params(s: str) -> list[tuple[str, str]]:
    s = s.strip()
    if not s:
        return []
    params = []
    for chunk in s.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        chunk = re.sub(r'\s*=\s*[^,]+$', '', chunk)  # strip default values
        m = re.match(rf'({_TYPE_PAT})\s+(\w+)', chunk)
        if not m:
            raise ValueError(f"Cannot parse parameter: {chunk!r}")
        params.append((_norm_type(m.group(1)), m.group(2)))
    return params


def _find_functions(code: str) -> list[tuple[str, str, list]]:
    results = []
    for m in _FUNC_RE.finditer(code):
        try:
            ret = _norm_type(m.group(1))
            name = m.group(2)
            params = _parse_params(m.group(3))
            results.append((ret, name, params))
        except ValueError:
            continue
    return results


# ---------------------------------------------------------------------------
# Extern "C" wrapper generation
# ---------------------------------------------------------------------------
def _gen_wrapper(ret: str, name: str, params: list[tuple[str, str]]) -> str:
    c_params = ['double* __out', 'int* __n_out']
    setup: list[str] = []
    call_args: list[str] = []

    for ptype, pname in params:
        if ptype == 'int':
            c_params.append(f'int {pname}')
            call_args.append(pname)
        elif ptype == 'double':
            c_params.append(f'double {pname}')
            call_args.append(pname)
        elif ptype == 'vec':
            c_params += [f'double* {pname}_data', f'int {pname}_len']
            setup.append(
                f'std::vector<double> {pname}({pname}_data, {pname}_data + {pname}_len);'
            )
            call_args.append(pname)
        elif ptype == 'ivec':
            c_params += [f'int* {pname}_data', f'int {pname}_len']
            setup.append(
                f'std::vector<int> {pname}({pname}_data, {pname}_data + {pname}_len);'
            )
            call_args.append(pname)
        elif ptype == 'avec':
            # arma::vec shares the buffer directly (no copy, faster)
            c_params += [f'double* {pname}_data', f'int {pname}_len']
            setup.append(
                f'arma::vec {pname}({pname}_data, (arma::uword){pname}_len, false, true);'
            )
            call_args.append(pname)
        elif ptype == 'amat':
            # numpy is row-major; Armadillo is col-major.
            # Construct as (cols×rows) then transpose → correct (rows×cols) matrix.
            c_params += [f'double* {pname}_data', f'int {pname}_rows', f'int {pname}_cols']
            setup.append(
                f'arma::mat {pname}('
                f'arma::mat({pname}_data, (arma::uword){pname}_cols, '
                f'(arma::uword){pname}_rows, false, true).t());'
            )
            call_args.append(pname)

    call = f'{name}({", ".join(call_args)})'
    if ret == 'vec':
        store = (
            f'auto __r = {call};\n'
            f'            *__n_out = (int)__r.size();\n'
            f'            for (int __i = 0; __i < *__n_out; __i++) __out[__i] = __r[__i];'
        )
    elif ret == 'avec':
        # arma::vec — elements stored contiguously, just memcpy via loop
        store = (
            f'auto __r = {call};\n'
            f'            *__n_out = (int)__r.n_elem;\n'
            f'            std::copy(__r.memptr(), __r.memptr() + __r.n_elem, __out);'
        )
    elif ret == 'amat':
        # arma::mat is col-major internally; copy out in row-major order for numpy
        store = (
            f'auto __r = {call};\n'
            f'            *__n_out = (int)(__r.n_rows * __r.n_cols);\n'
            f'            int __idx = 0;\n'
            f'            for (arma::uword __i = 0; __i < __r.n_rows; __i++)\n'
            f'                for (arma::uword __j = 0; __j < __r.n_cols; __j++)\n'
            f'                    __out[__idx++] = __r(__i, __j);'
        )
    else:
        store = f'__out[0] = (double)({call});\n            *__n_out = 1;'

    sig = ', '.join(c_params)
    setup_block = '\n            '.join(setup)

    return (
        f'extern "C" {{\n'
        f'    SCPP_EXPORT int scpp_{name}({sig}) {{\n'
        f'        try {{\n'
        f'            {setup_block}\n'
        f'            {store}\n'
        f'            return 0;\n'
        f'        }} catch (...) {{ *__n_out = 0; return 1; }}\n'
        f'    }}\n'
        f'}}\n'
    )


def _build_source(user_code: str, funcs: list, use_arma: bool = False) -> str:
    if '#include' in user_code:
        includes = ''
    else:
        includes = _DEFAULT_INCLUDES
        if use_arma and '#include <armadillo>' not in user_code:
            includes += '#include <armadillo>\n'
    wrappers = '\n'.join(_gen_wrapper(r, n, p) for r, n, p in funcs)
    return _EXPORT_MACRO + '\n' + includes + '\n' + user_code + '\n\n' + wrappers


# ---------------------------------------------------------------------------
# Compile, load, cache
# ---------------------------------------------------------------------------
def _load_and_cache(key: str, src: str,
                    extra_flags: list[str] | None = None) -> ctypes.CDLL:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _, kind = _find_compiler()
    lib_path = _CACHE_DIR / f'{key}{_lib_suffix(kind)}'
    src_path = _CACHE_DIR / f'{key}.cpp'
    if not lib_path.exists():
        _compile(src, lib_path, src_path, extra_flags)
    lib = ctypes.CDLL(str(lib_path))
    _lib_cache[key] = lib
    return lib


# ---------------------------------------------------------------------------
# Python callable builder
# ---------------------------------------------------------------------------
def _make_callable(
    lib: ctypes.CDLL,
    name: str,
    ret: str,
    params: list[tuple[str, str]],
) -> Callable:
    fn = getattr(lib, f'scpp_{name}')
    fn.restype = ctypes.c_int
    argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
    for ptype, _ in params:
        if ptype == 'int':
            argtypes.append(ctypes.c_int)
        elif ptype == 'double':
            argtypes.append(ctypes.c_double)
        elif ptype in ('vec', 'avec'):
            argtypes += [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        elif ptype == 'ivec':
            argtypes += [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        elif ptype == 'amat':
            # data pointer, rows, cols
            argtypes += [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
    fn.argtypes = argtypes

    def _call(*args):
        buf = (ctypes.c_double * _MAX_RESULT)()
        n_out = ctypes.c_int(0)
        call_args: list = [buf, ctypes.byref(n_out)]
        for (ptype, _), val in zip(params, args):
            if ptype == 'int':
                call_args.append(ctypes.c_int(int(val)))
            elif ptype == 'double':
                call_args.append(ctypes.c_double(float(val)))
            elif ptype in ('vec', 'avec'):
                # 2D arrays are flattened row-major (C order)
                arr = np.asarray(val, dtype=np.float64)
                if arr.ndim == 2:
                    arr = np.ascontiguousarray(arr.flatten())
                elif arr.ndim != 1:
                    raise TypeError(f"Expected 1D or 2D array, got {arr.ndim}D")
                else:
                    arr = np.ascontiguousarray(arr)
                call_args.append(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
                call_args.append(ctypes.c_int(arr.size))
            elif ptype == 'ivec':
                arr = np.asarray(val, dtype=np.int32)
                if arr.ndim != 1:
                    raise TypeError(f"std::vector<int> requires a 1D array, got {arr.ndim}D")
                arr = np.ascontiguousarray(arr)
                call_args.append(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
                call_args.append(ctypes.c_int(arr.size))
            elif ptype == 'amat':
                # Always send as contiguous row-major (C order); wrapper transposes
                arr = np.asarray(val, dtype=np.float64)
                if arr.ndim != 2:
                    raise TypeError(f"arma::mat requires a 2D array, got {arr.ndim}D")
                arr = np.ascontiguousarray(arr)  # ensures C order
                call_args.append(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
                call_args.append(ctypes.c_int(arr.shape[0]))
                call_args.append(ctypes.c_int(arr.shape[1]))
        rc = fn(*call_args)
        if rc != 0:
            raise RuntimeError(f"Runtime error in {name!r} (code {rc})")
        return np.frombuffer(buf, dtype=np.float64, count=n_out.value).copy()

    _call.__name__ = name
    return _call


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def cppFunction(code: str) -> 'Callable | dict[str, Callable]':
    """Compile one or more C++ functions and return Python callable(s).

    Supported parameter / return types: int, double, std::vector<double>.
    Standard headers (<vector>, <cmath>, <random>, …) are always available.

    Returns a single callable when one function is found, or a dict
    {name: callable} when multiple functions are defined.

    Examples
    --------
    rnorm = cppFunction('''
        std::vector<double> rnorm(int n, double mean, double sd) {
            thread_local std::mt19937_64 gen{std::random_device{}()};
            std::normal_distribution<double> dist(mean, sd);
            std::vector<double> out(n);
            for (auto& v : out) v = dist(gen);
            return out;
        }
    ''')
    result = rnorm(10, 0.0, 1.0)   # → numpy array
    """
    funcs = _find_functions(code)
    if not funcs:
        raise ValueError(
            "No function definitions with supported types found. "
            "Supported types: int, double, std::vector<double>, std::vector<int>, "
            "arma::vec, arma::mat."
        )
    use_arma = _needs_armadillo(code, funcs)
    key = hashlib.md5((_VERSION + code.strip()).encode()).hexdigest()
    if key not in _lib_cache:
        src = _build_source(code, funcs, use_arma=use_arma)
        extra: list[str] = []
        if use_arma:
            cflags, ldflags = _find_armadillo()
            extra = cflags + ldflags
        _load_and_cache(key, src, extra_flags=extra)
    lib = _lib_cache[key]
    callables = {
        name: _make_callable(lib, name, ret, params)
        for ret, name, params in funcs
    }
    return next(iter(callables.values())) if len(callables) == 1 else callables


def sourceCpp(path: 'str | Path') -> 'Callable | dict[str, Callable]':
    """Compile a .cpp file and return Python callable(s)."""
    return cppFunction(Path(path).read_text(encoding="utf-8"))


def evalCpp(expr: str) -> np.ndarray:
    """Evaluate a single C++ expression and return a numpy array.

    The expression must produce a double or std::vector<double>.

    Example
    -------
    evalCpp("std::sqrt(2.0)")   # → array([1.41421356])
    """
    src = (
        _EXPORT_MACRO + '\n' +
        _DEFAULT_INCLUDES + '\n'
        'namespace _scpp_eval {\n'
        '    inline std::vector<double> to_vec(std::vector<double> v) { return v; }\n'
        '    inline std::vector<double> to_vec(double x) { return {x}; }\n'
        '}\n'
        'extern "C" {\n'
        '    SCPP_EXPORT int scpp__eval_(double* __out, int* __n_out) {\n'
        '        try {\n'
        '            auto __r = _scpp_eval::to_vec((' + expr + '));\n'
        '            *__n_out = (int)__r.size();\n'
        '            for (int __i = 0; __i < *__n_out; __i++) __out[__i] = __r[__i];\n'
        '            return 0;\n'
        '        } catch (...) { *__n_out = 0; return 1; }\n'
        '    }\n'
        '}\n'
    )
    key = hashlib.md5((_VERSION + 'evalCpp:' + expr.strip()).encode()).hexdigest()
    if key not in _lib_cache:
        _load_and_cache(key, src)
    lib = _lib_cache[key]
    fn = lib.scpp__eval_
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
    buf = (ctypes.c_double * _MAX_RESULT)()
    n_out = ctypes.c_int(0)
    rc = fn(buf, ctypes.byref(n_out))
    if rc != 0:
        raise RuntimeError(f"Runtime error evaluating: {expr!r}")
    return np.frombuffer(buf, dtype=np.float64, count=n_out.value).copy()


def check() -> None:
    """Print a diagnostics summary: Python, platform, compiler, and a smoke test."""
    print(f"statscpp {_VERSION}")
    print(f"Python   {sys.version.split()[0]}  ({sys.executable})")
    print(f"Platform {platform.system()} {platform.machine()}")
    print(f"Cache    {_CACHE_DIR}")

    try:
        compiler, kind = _find_compiler()
        # Ask the compiler for its version
        flag = "/?" if kind == "msvc" else "--version"
        r = subprocess.run([compiler, flag], capture_output=True, text=True)
        ver_line = (r.stdout or r.stderr).splitlines()[0] if (r.stdout or r.stderr) else ""
        print(f"Compiler {compiler}  [{ver_line.strip()}]")
    except RuntimeError as e:
        print(f"Compiler NOT FOUND — {e}")
        return

    # Armadillo
    try:
        cflags, ldflags = _find_armadillo()
        print(f"Arma     found  (flags: {' '.join(cflags + ldflags)})")
    except RuntimeError as e:
        print(f"Arma     NOT FOUND — {e}")

    # Smoke test
    try:
        result = evalCpp("std::sqrt(2.0)")
        ok = abs(result[0] - 1.4142135623730951) < 1e-10
        print(f"Smoke    evalCpp('std::sqrt(2.0)') = {result[0]:.6f}  {'OK' if ok else 'WRONG'}")
    except Exception as e:
        print(f"Smoke    FAILED — {e}")
