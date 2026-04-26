"""
Type system for statscpp.

This module defines the "canonical tags" for supported C++ types. Think of it
as a universal language the entire pipeline uses:

  - Python function receives code with C++ types
  - Parser converts C++ types to canonical tags ("vec", "int", etc.)
  - Code generator and marshaler use tags to create glue code
  - Finally, the right conversions happen between Python and C++

Supported types:
  INT    ↔ int, long, size_t (Python int)
  DOUBLE ↔ double, float (Python float)
  VEC    ↔ std::vector<double> (NumPy 1-D array, float64)
  IVEC   ↔ std::vector<int> (NumPy 1-D array, int32)
  AVEC   ↔ arma::vec (NumPy 1-D array via Armadillo)
  AMAT   ↔ arma::mat (NumPy 2-D array via Armadillo)

The key insight: by using canonical tags, we separate concerns. The parser
worries about C++ syntax, the generator worries about code generation,
and the marshaler worries about Python ↔ C conversion. Each module is simple.
"""
import re

# ---------------------------------------------------------------------------
# Canonical type tags
# ---------------------------------------------------------------------------
INT    = "int"    # int / long / size_t
DOUBLE = "double" # double / float
VEC    = "vec"    # std::vector<double>
IVEC   = "ivec"   # std::vector<int>
AVEC   = "avec"   # arma::vec
AMAT   = "amat"   # arma::mat

# All supported tags, for external reference
ALL_TAGS = (INT, DOUBLE, VEC, IVEC, AVEC, AMAT)

# ---------------------------------------------------------------------------
# Regex fragment that matches any supported C++ type (used by parser)
# ---------------------------------------------------------------------------
PATTERN = (
    r'(?:'
    r'(?:std::)?vector\s*<\s*double\s*>'
    r'|(?:std::)?vector\s*<\s*int\s*>'
    r'|arma::mat'   # must come before arma::vec to avoid prefix match
    r'|arma::vec'
    r'|double|float|int|long|size_t'
    r')'
)


def normalize(raw: str) -> str:
    """Map a raw C++ type string to a canonical tag.

    Raises ValueError for unsupported types, with a helpful message.
    """
    t = re.sub(r'\s+', ' ', raw.strip())
    if re.search(r'vector\s*<\s*double\s*>', t): return VEC
    if re.search(r'vector\s*<\s*int\s*>',    t): return IVEC
    if t == 'arma::vec':                          return AVEC
    if t == 'arma::mat':                          return AMAT
    if t in ('double', 'float'):                  return DOUBLE
    if t in ('int', 'long', 'size_t'):            return INT
    
    # Help user understand what went wrong
    raise ValueError(
        f"❌ Unsupported return type or parameter type: {t!r}\n\n"
        f"Supported types:\n"
        f"  • int, long, size_t           → Python int\n"
        f"  • double, float               → Python float\n"
        f"  • std::vector<double>         → NumPy 1D array\n"
        f"  • std::vector<int>            → NumPy 1D integer array\n"
        f"  • arma::vec                   → NumPy 1D array (Armadillo)\n"
        f"  • arma::mat                   → NumPy 2D array (Armadillo)\n\n"
        f"Tip: Helper functions (e.g., with internal types like std::string)\n"
        f"are ignored and won't cause errors."
    )
