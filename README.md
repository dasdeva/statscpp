# statscpp

Write C++ statistical code in Python strings — compile and run it.

Like [Rcpp](https://www.rcpp.org/) but called from Python.

```python
from statscpp import cppFunction

rnorm = cppFunction('''
    std::vector<double> rnorm(int n, double mean, double sd) {
        thread_local std::mt19937_64 gen{std::random_device{}()};
        std::normal_distribution<double> dist(mean, sd);
        std::vector<double> out(n);
        for (auto& v : out) v = dist(gen);
        return out;
    }
''')

rnorm(10, 0.0, 1.0)   # → numpy array of 10 floats
```

C++ is compiled once per unique code string, cached to `~/.cache/statscpp/`, and reused across sessions.

---

## Installation

### Step 1 — install a C++ compiler

statscpp needs a C++17 compiler on your PATH. Install one for your platform:

| Platform | Command |
|---|---|
| **macOS** | `xcode-select --install` |
| **Ubuntu / Debian** | `sudo apt install build-essential` |
| **Fedora / RHEL** | `sudo dnf install gcc-c++` |
| **Arch** | `sudo pacman -S base-devel` |
| **WSL** | same as Ubuntu above |
| **Windows** | Install [MinGW-w64](https://www.mingw-w64.org/) via `winget install MSYS2.MSYS2`, then add `g++` to PATH. Alternatively install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) (includes `cl.exe`). |

### Step 2 — install statscpp

```bash
pip install statscpp
```

### Verify your setup

```python
import statscpp
statscpp.check()
```

Expected output (example on macOS):
```
statscpp 0.1.0
Python   3.12.3  (/usr/local/bin/python3)
Platform Darwin arm64
Cache    /Users/you/.cache/statscpp
Compiler /usr/bin/g++  [Apple clang version 15.0.0]
Smoke    evalCpp('std::sqrt(2.0)') = 1.414214  OK
```

---

## API

### `cppFunction(code)` — compile a function, get a callable

```python
from statscpp import cppFunction

# Single function → returns a callable
dnorm = cppFunction('''
    double dnorm(double x, double mean, double sd) {
        const double PI = std::acos(-1.0);
        double z = (x - mean) / sd;
        return std::exp(-0.5 * z * z) / (sd * std::sqrt(2.0 * PI));
    }
''')
dnorm(0.0, 0.0, 1.0)   # → array([0.39894228])

# Multiple functions → returns a dict {name: callable}
fns = cppFunction('''
    std::vector<double> scale(std::vector<double> x, double f) {
        for (auto& v : x) v *= f;
        return x;
    }
    double vsum(std::vector<double> x) {
        return std::accumulate(x.begin(), x.end(), 0.0);
    }
''')
fns['scale']([1.0, 2.0, 3.0], 2.0)   # → array([2., 4., 6.])
fns['vsum']([1.0, 2.0, 3.0])          # → array([6.])
```

**Supported types** for parameters and return values:

| C++ type | Python / numpy |
|---|---|
| `int` | `int` |
| `double` | `float` |
| `std::vector<double>` | `numpy.ndarray` (float64) |

All return values come back as `numpy.ndarray`. Scalar-returning functions give a 1-element array.

Standard headers are always available without `#include`:
`<vector>`, `<cmath>`, `<random>`, `<algorithm>`, `<numeric>`, `<stdexcept>`, `<string>`

---

### `sourceCpp(path)` — compile from a file

```python
from statscpp import sourceCpp
fns = sourceCpp("my_models.cpp")
```

---

### `evalCpp(expr)` — evaluate a one-liner

```python
from statscpp import evalCpp
evalCpp("std::sqrt(2.0)")       # → array([1.41421356])
evalCpp("std::tgamma(6.0)")     # → array([120.])  (5!)
```

---

### `check()` — verify environment

```python
import statscpp
statscpp.check()
```

---

## How it works

1. Your C++ string is parsed to find function signatures (return type, name, parameters).
2. A thin `extern "C"` wrapper is generated around each function that marshals data through flat double arrays.
3. The combined source is compiled to a shared library (`.so` / `.dylib` / `.dll`) with `g++` or `cl.exe`.
4. The library is loaded with Python's `ctypes` and the wrapped functions are returned as Python callables.
5. The compiled library is cached by content hash in `~/.cache/statscpp/` — subsequent calls with the same code string are instant.

---

## License

GNU GPL v3 — see [LICENSE](LICENSE).
