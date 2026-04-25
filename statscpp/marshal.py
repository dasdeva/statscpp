"""
Python ↔ C type marshaling.

Converts Python / numpy values into the flat C arguments expected by the
generated extern "C" shims, and wraps a compiled shim in a Python callable
that returns numpy arrays.

Memory-safety note
------------------
For array parameters, ``_to_c`` returns both the ctypes objects *and* the
underlying numpy array.  The caller must keep the numpy array alive until
after the C function returns; storing it in a ``refs`` list inside ``_call``
achieves this without relying on CPython's reference-counting timing.
"""
import ctypes
from typing import Callable

import numpy as np

from . import types

MAX_RESULT: int = 1_000_000   # max doubles in the output buffer (≈ 8 MB)

# Convenience aliases
_Pd = ctypes.POINTER(ctypes.c_double)
_Pi = ctypes.POINTER(ctypes.c_int)


# ---------------------------------------------------------------------------
# ctypes argument-type list
# ---------------------------------------------------------------------------

def argtypes_for(params: list[tuple[str, str]]) -> list:
    """Build the ctypes argtypes list for a shim's parameter signature."""
    result = [_Pd, _Pi]   # always: double* __out, int* __n_out
    for tag, _ in params:
        if tag == types.INT:
            result.append(ctypes.c_int)
        elif tag == types.DOUBLE:
            result.append(ctypes.c_double)
        elif tag in (types.VEC, types.AVEC):
            result += [_Pd, ctypes.c_int]
        elif tag == types.IVEC:
            result += [_Pi, ctypes.c_int]
        elif tag == types.AMAT:
            result += [_Pd, ctypes.c_int, ctypes.c_int]
    return result


# ---------------------------------------------------------------------------
# Python value → C arguments
# ---------------------------------------------------------------------------

def _to_c(tag: str, val) -> tuple[list, object | None]:
    """Convert one Python value to C call arguments.

    Returns (c_args, ref) where:
      - c_args  is a list of ctypes-compatible values to append to call_args
      - ref     is the numpy array that must stay alive during the C call,
                or None for scalar types
    """
    if tag == types.INT:
        return [ctypes.c_int(int(val))], None

    if tag == types.DOUBLE:
        return [ctypes.c_double(float(val))], None

    if tag in (types.VEC, types.AVEC):
        arr = np.asarray(val, dtype=np.float64)
        if arr.ndim == 2:
            arr = np.ascontiguousarray(arr.flatten())
        elif arr.ndim == 1:
            arr = np.ascontiguousarray(arr)
        else:
            raise TypeError(f"Expected 1D or 2D array, got {arr.ndim}D")
        return [arr.ctypes.data_as(_Pd), ctypes.c_int(arr.size)], arr

    if tag == types.IVEC:
        arr = np.asarray(val, dtype=np.int32)
        if arr.ndim != 1:
            raise TypeError(f"std::vector<int> requires a 1D array, got {arr.ndim}D")
        arr = np.ascontiguousarray(arr)
        return [arr.ctypes.data_as(_Pi), ctypes.c_int(arr.size)], arr

    if tag == types.AMAT:
        arr = np.asarray(val, dtype=np.float64)
        if arr.ndim != 2:
            raise TypeError(f"arma::mat requires a 2D array, got {arr.ndim}D")
        arr = np.ascontiguousarray(arr)   # C order (row-major); wrapper handles transpose
        return [arr.ctypes.data_as(_Pd),
                ctypes.c_int(arr.shape[0]),
                ctypes.c_int(arr.shape[1])], arr

    raise TypeError(f"Unknown type tag: {tag!r}")


# ---------------------------------------------------------------------------
# Public: build a Python callable around a compiled shim
# ---------------------------------------------------------------------------

def make_callable(
    lib: ctypes.CDLL,
    name: str,
    ret: str,
    params: list[tuple[str, str]],
) -> Callable:
    """Wrap the extern-C shim ``scpp_{name}`` in a Python function.

    The returned callable accepts Python / numpy arguments and returns a
    ``numpy.ndarray`` (always 1-D float64).
    """
    fn = getattr(lib, f"scpp_{name}")
    fn.restype  = ctypes.c_int
    fn.argtypes = argtypes_for(params)

    def _call(*args):
        buf   = (ctypes.c_double * MAX_RESULT)()
        n_out = ctypes.c_int(0)

        call_args: list = [buf, ctypes.byref(n_out)]
        refs: list = []   # keep numpy arrays alive until fn() returns

        for (tag, _), val in zip(params, args):
            c_args, ref = _to_c(tag, val)
            call_args.extend(c_args)
            if ref is not None:
                refs.append(ref)

        rc = fn(*call_args)
        # refs is still in scope here, preventing premature GC of array buffers

        if rc != 0:
            raise RuntimeError(f"Runtime error in {name!r} (code {rc})")
        return np.frombuffer(buf, dtype=np.float64, count=n_out.value).copy()

    _call.__name__ = name
    return _call
