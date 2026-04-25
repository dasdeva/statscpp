"""
Type system for statscpp.

Each supported C++ type maps to a short canonical tag used throughout the
pipeline (parsing → wrapper generation → marshaling).
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

    Raises ValueError for unsupported types.
    """
    t = re.sub(r'\s+', ' ', raw.strip())
    if re.search(r'vector\s*<\s*double\s*>', t): return VEC
    if re.search(r'vector\s*<\s*int\s*>',    t): return IVEC
    if t == 'arma::vec':                          return AVEC
    if t == 'arma::mat':                          return AMAT
    if t in ('double', 'float'):                  return DOUBLE
    if t in ('int', 'long', 'size_t'):            return INT
    raise ValueError(
        f"Unsupported type {t!r}. "
        f"Supported: int, double, std::vector<double>, std::vector<int>, "
        f"arma::vec, arma::mat."
    )
