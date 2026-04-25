"""
Public API functions.

This module wires together the pipeline:
  parser → wrapper → cache (→ compiler) → marshal → callable
"""
import ctypes
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from . import armadillo, cache, compiler, marshal, parser, types, wrapper
from ._version import __version__


def cppFunction(code: str) -> "Callable | dict[str, Callable]":
    """Compile C++ function(s) in *code* and return Python callable(s).

    Supported parameter/return types
    ---------------------------------
    - ``int``                   ↔  Python int
    - ``double``                ↔  Python float
    - ``std::vector<double>``   ↔  1-D or 2-D numpy array (float64)
    - ``std::vector<int>``      ↔  1-D numpy array (int32)
    - ``arma::vec``             ↔  1-D numpy array (float64)
    - ``arma::mat``             ↔  2-D numpy array (float64, row-major)

    Standard headers (``<vector>``, ``<cmath>``, ``<random>``, …) are
    automatically included.  ``#include <armadillo>`` is added when
    ``arma::`` appears in *code*.

    Returns a single callable for one function, or a ``{name: callable}``
    dict for multiple functions.

    Examples
    --------
    >>> rnorm = cppFunction('''
    ...     std::vector<double> rnorm(int n, double mean, double sd) {
    ...         thread_local std::mt19937_64 gen{std::random_device{}()};
    ...         std::normal_distribution<double> dist(mean, sd);
    ...         std::vector<double> out(n);
    ...         for (auto& v : out) v = dist(gen);
    ...         return out;
    ...     }
    ... ''')
    >>> rnorm(10, 0.0, 1.0)   # → numpy array of 10 floats
    """
    funcs = parser.find_functions(code)
    if not funcs:
        raise ValueError(
            "No function definitions with supported types found in the provided code.\n"
            f"Supported types: {', '.join(types.ALL_TAGS)}."
        )

    use_arma   = armadillo.needed(code, funcs)
    key        = cache.make_key(code)
    src        = wrapper.build_source(code, funcs, use_arma=use_arma)
    extra: list[str] = []
    if use_arma:
        cflags, ldflags = armadillo.find()
        extra = cflags + ldflags

    lib = cache.load(key, src, extra_flags=extra)
    callables = {
        name: marshal.make_callable(lib, name, ret, params)
        for ret, name, params in funcs
    }
    return next(iter(callables.values())) if len(callables) == 1 else callables


def sourceCpp(path: "str | Path") -> "Callable | dict[str, Callable]":
    """Compile a ``.cpp`` file and return Python callable(s).

    Equivalent to ``cppFunction(Path(path).read_text())``.
    """
    return cppFunction(Path(path).read_text(encoding="utf-8"))


def evalCpp(expr: str) -> np.ndarray:
    """Evaluate a single C++ expression and return a numpy array.

    The expression must produce a ``double`` or ``std::vector<double>``.

    Examples
    --------
    >>> evalCpp("std::sqrt(2.0)")    # → array([1.41421356])
    >>> evalCpp("std::tgamma(6.0)")  # → array([120.])   (5!)
    """
    # Build source manually: the _to_vec helpers must not be picked up by
    # parser.find_functions, so they live in a private namespace.
    src = (
        wrapper.EXPORT_MACRO + "\n"
        + wrapper.DEFAULT_INCLUDES + "\n"
        "namespace _scpp_eval {\n"
        "    inline std::vector<double> to_vec(std::vector<double> v) { return v; }\n"
        "    inline std::vector<double> to_vec(double x) { return {x}; }\n"
        "}\n"
        "extern \"C\" {\n"
        "    SCPP_EXPORT int scpp__eval_(double* __out, int* __n_out) {\n"
        "        try {\n"
        f"            auto __r = _scpp_eval::to_vec(({expr}));\n"
        "            *__n_out = (int)__r.size();\n"
        "            for (int __i = 0; __i < *__n_out; __i++) __out[__i] = __r[__i];\n"
        "            return 0;\n"
        "        } catch (...) { *__n_out = 0; return 1; }\n"
        "    }\n"
        "}\n"
    )
    key = cache.make_key("evalCpp:" + expr)
    lib = cache.load(key, src)

    fn            = lib.scpp__eval_
    fn.restype    = ctypes.c_int
    fn.argtypes   = [marshal._Pd, marshal._Pi]

    buf   = (ctypes.c_double * marshal.MAX_RESULT)()
    n_out = ctypes.c_int(0)
    rc    = fn(buf, ctypes.byref(n_out))
    if rc != 0:
        raise RuntimeError(f"Runtime error evaluating: {expr!r}")
    return np.frombuffer(buf, dtype=np.float64, count=n_out.value).copy()


def check() -> None:
    """Print an environment diagnostics report.

    Reports Python version, platform, compiler, Armadillo, and runs a
    quick smoke test.  Useful for confirming a working install.
    """
    print(f"statscpp {__version__}")
    print(f"Python   {sys.version.split()[0]}  ({sys.executable})")
    print(f"Platform {platform.system()} {platform.machine()}")
    print(f"Cache    {cache.CACHE_DIR}")

    # Compiler
    try:
        path, kind = compiler.find()
        flag = "/?" if kind == "msvc" else "--version"
        r = subprocess.run([path, flag], capture_output=True, text=True)
        ver = (r.stdout or r.stderr).splitlines()[0].strip() if (r.stdout or r.stderr) else ""
        print(f"Compiler {path}  [{ver}]")
    except RuntimeError as exc:
        print(f"Compiler NOT FOUND — {exc}")
        return

    # Armadillo (optional)
    try:
        cflags, ldflags = armadillo.find()
        print(f"Arma     found  (flags: {' '.join(cflags + ldflags)})")
    except RuntimeError:
        print("Arma     not installed  "
              "(brew install armadillo / apt install libarmadillo-dev)")

    # Smoke test
    try:
        result = evalCpp("std::sqrt(2.0)")
        ok = abs(result[0] - 1.4142135623730951) < 1e-10
        print(f"Smoke    evalCpp('std::sqrt(2.0)') = {result[0]:.6f}  "
              f"{'OK' if ok else 'WRONG'}")
    except Exception as exc:
        print(f"Smoke    FAILED — {exc}")
