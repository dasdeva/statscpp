"""
C++ function signature parser.

Extracts function definitions from user-supplied code strings so the rest of
the pipeline knows what extern "C" wrappers to generate.
"""
import re
from . import types

# Matches:  <return_type> <name>(<params>) {
# Handles multi-line parameter lists and optional trailing `const`.
_FUNC_RE = re.compile(
    rf'({types.PATTERN})\s+(\w+)\s*\(([^{{;]*?)\)\s*(?:const\s*)?{{',
    re.MULTILINE | re.DOTALL,
)


def parse_params(raw: str) -> list[tuple[str, str]]:
    """Parse a C++ parameter list string into [(tag, name), ...].

    Strips default values.  Raises ValueError on unrecognised types.
    """
    raw = raw.strip()
    if not raw:
        return []
    result = []
    for chunk in raw.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        chunk = re.sub(r'\s*=\s*[^,]+$', '', chunk)   # drop default value
        m = re.match(rf'({types.PATTERN})\s+(\w+)', chunk)
        if not m:
            raise ValueError(f"Cannot parse parameter: {chunk!r}")
        result.append((types.normalize(m.group(1)), m.group(2)))
    return result


def find_functions(code: str) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """Return [(ret_tag, name, params), ...] for every parseable function in code.

    Functions whose return type or any parameter type is unsupported are
    silently skipped so that helper functions with internal types don't
    break compilation.
    """
    results = []
    for m in _FUNC_RE.finditer(code):
        try:
            ret    = types.normalize(m.group(1))
            name   = m.group(2)
            params = parse_params(m.group(3))
            results.append((ret, name, params))
        except ValueError:
            continue
    return results
