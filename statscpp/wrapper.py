"""
C++ source generation.

Builds the full .cpp file that gets compiled: the DLL-export macro, standard
includes, the user's code verbatim, and one extern "C" shim per function.
"""
from . import types

# ---------------------------------------------------------------------------
# Fixed source fragments
# ---------------------------------------------------------------------------
EXPORT_MACRO = """\
#ifdef _WIN32
#  define SCPP_EXPORT __declspec(dllexport)
#else
#  define SCPP_EXPORT
#endif
"""

DEFAULT_INCLUDES = """\
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
"""

ARMA_INCLUDE = "#include <armadillo>\n"


# ---------------------------------------------------------------------------
# Per-type helpers — each returns the C-level declarations / setup needed
# ---------------------------------------------------------------------------

def _c_params(ptype: str, pname: str) -> list[str]:
    """C-level parameter declarations for one logical parameter."""
    if ptype == types.INT:    return [f'int {pname}']
    if ptype == types.DOUBLE: return [f'double {pname}']
    if ptype == types.VEC:    return [f'double* {pname}_data', f'int {pname}_len']
    if ptype == types.IVEC:   return [f'int* {pname}_data',    f'int {pname}_len']
    if ptype == types.AVEC:   return [f'double* {pname}_data', f'int {pname}_len']
    if ptype == types.AMAT:   return [f'double* {pname}_data', f'int {pname}_rows', f'int {pname}_cols']
    raise ValueError(f"Unknown type tag: {ptype!r}")


def _setup(ptype: str, pname: str) -> str:
    """C++ statement that reconstructs the user-facing type from C params.

    Returns an empty string for scalars (no setup needed).
    """
    if ptype == types.VEC:
        return f'std::vector<double> {pname}({pname}_data, {pname}_data + {pname}_len);'
    if ptype == types.IVEC:
        return f'std::vector<int> {pname}({pname}_data, {pname}_data + {pname}_len);'
    if ptype == types.AVEC:
        # Share the caller's buffer directly — no copy, the buffer outlives this call.
        return f'arma::vec {pname}({pname}_data, (arma::uword){pname}_len, false, true);'
    if ptype == types.AMAT:
        # numpy is row-major; Armadillo is col-major.
        # Interpret flat row-major data as a (cols × rows) col-major matrix,
        # then transpose to get the correct (rows × cols) matrix.
        return (
            f'arma::mat {pname}('
            f'arma::mat({pname}_data, (arma::uword){pname}_cols, '
            f'(arma::uword){pname}_rows, false, true).t());'
        )
    return ''  # INT, DOUBLE — passed directly, no setup


def _store(ret: str, call: str) -> str:
    """C++ statements that call the function and write the result to __out."""
    if ret == types.VEC:
        return (
            f'auto __r = {call};\n'
            f'            *__n_out = (int)__r.size();\n'
            f'            for (int __i = 0; __i < *__n_out; __i++) __out[__i] = __r[__i];'
        )
    if ret == types.AVEC:
        return (
            f'auto __r = {call};\n'
            f'            *__n_out = (int)__r.n_elem;\n'
            f'            std::copy(__r.memptr(), __r.memptr() + __r.n_elem, __out);'
        )
    if ret == types.AMAT:
        # Copy col-major Armadillo matrix out in row-major order for numpy.
        return (
            f'auto __r = {call};\n'
            f'            *__n_out = (int)(__r.n_rows * __r.n_cols);\n'
            f'            int __idx = 0;\n'
            f'            for (arma::uword __i = 0; __i < __r.n_rows; __i++)\n'
            f'                for (arma::uword __j = 0; __j < __r.n_cols; __j++)\n'
            f'                    __out[__idx++] = __r(__i, __j);'
        )
    # INT or DOUBLE — cast to double and return as a single-element array
    return f'__out[0] = (double)({call});\n            *__n_out = 1;'


# ---------------------------------------------------------------------------
# Public: wrapper and source assembly
# ---------------------------------------------------------------------------

def gen_wrapper(ret: str, name: str, params: list[tuple[str, str]]) -> str:
    """Generate a flat extern "C" shim for one user function."""
    c_param_list = ['double* __out', 'int* __n_out']
    setup_lines: list[str] = []
    call_args:   list[str] = []

    for ptype, pname in params:
        c_param_list.extend(_c_params(ptype, pname))
        stmt = _setup(ptype, pname)
        if stmt:
            setup_lines.append(stmt)
        call_args.append(pname)   # name is always the call argument

    call        = f'{name}({", ".join(call_args)})'
    store_block = _store(ret, call)
    sig         = ', '.join(c_param_list)
    setup_block = '\n            '.join(setup_lines)

    return (
        f'extern "C" {{\n'
        f'    SCPP_EXPORT int scpp_{name}({sig}) {{\n'
        f'        try {{\n'
        f'            {setup_block}\n'
        f'            {store_block}\n'
        f'            return 0;\n'
        f'        }} catch (...) {{ *__n_out = 0; return 1; }}\n'
        f'    }}\n'
        f'}}\n'
    )


def build_source(user_code: str, funcs: list, use_arma: bool = False) -> str:
    """Assemble the full .cpp: macros + includes + user code + shim wrappers."""
    if '#include' in user_code:
        # User manages their own includes; don't add duplicates.
        includes = ''
    else:
        includes = DEFAULT_INCLUDES
        if use_arma:
            includes += ARMA_INCLUDE

    wrappers = '\n'.join(gen_wrapper(r, n, p) for r, n, p in funcs)
    return EXPORT_MACRO + '\n' + includes + '\n' + user_code + '\n\n' + wrappers
