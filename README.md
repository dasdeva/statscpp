# statscpp

Write C++ statistical code in Python strings — compile and run it.
The C++ compiles once, caches, and returns numpy arrays.

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

---

## Installation

### 1. Install a C++ compiler

statscpp compiles your code at runtime and needs a C++17 compiler on PATH.

| Platform | Command |
|---|---|
| **macOS** | `xcode-select --install` |
| **Ubuntu / Debian** | `sudo apt install build-essential` |
| **Fedora / RHEL** | `sudo dnf install gcc-c++` |
| **Arch Linux** | `sudo pacman -S base-devel` |
| **WSL** | same as Ubuntu |
| **Windows** | [MinGW-w64](https://www.mingw-w64.org/): `winget install MSYS2.MSYS2` then add `g++` to PATH, **or** install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) (`cl.exe` is auto-detected) |

### 2. Install statscpp

```bash
pip install statscpp
```

### 3. (Optional) Install Armadillo for matrix operations

| Platform | Command |
|---|---|
| **macOS** | `brew install armadillo` |
| **Ubuntu / Debian** | `sudo apt install libarmadillo-dev` |
| **Fedora / RHEL** | `sudo dnf install armadillo-devel` |
| **Arch Linux** | `sudo pacman -S armadillo` |
| **Windows** | [vcpkg](https://vcpkg.io): `vcpkg install armadillo` |

### 4. Verify

```python
import statscpp
statscpp.check()
```

```
statscpp 0.1.0
Python   3.12.3  (/usr/bin/python3)
Platform Darwin arm64
Cache    /Users/you/.cache/statscpp
Compiler /usr/bin/g++  [Apple clang version 15.0.0]
Arma     found  (flags: -I/opt/homebrew/include -L/opt/homebrew/lib -larmadillo)
Smoke    evalCpp('std::sqrt(2.0)') = 1.414214  OK
```

---

## Quick start

```python
from statscpp import cppFunction, evalCpp, sourceCpp

# --- one-liner expressions ---
evalCpp("std::sqrt(2.0)")       # → array([1.41421356])
evalCpp("std::tgamma(6.0)")     # → array([120.])   (5!)

# --- define a function ---
dnorm = cppFunction('''
    double dnorm(double x, double mean, double sd) {
        const double PI = std::acos(-1.0);
        double z = (x - mean) / sd;
        return std::exp(-0.5 * z * z) / (sd * std::sqrt(2.0 * PI));
    }
''')
dnorm(0.0, 0.0, 1.0)    # → array([0.39894228])

# --- compile from a file ---
fns = sourceCpp("my_models.cpp")

# --- multiple functions in one block ---
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

---

## Armadillo

`arma::` anywhere in your code auto-triggers `#include <armadillo>` and the
right linker flags — no setup needed on your end.

```python
from statscpp import cppFunction
import numpy as np

inv = cppFunction('''
    arma::mat mat_inv(arma::mat A) { return arma::inv(A); }
''')

A     = np.array([[1., 2.], [3., 4.]])
A_inv = inv(A).reshape(2, 2)   # reshape the flat result back to 2-D
```

---

## Type reference

| C++ type | Accepted Python input | Return shape |
|---|---|---|
| `int` | `int` | 1-element array |
| `double` | `float` | 1-element array |
| `std::vector<double>` | list, 1-D or 2-D `ndarray` (float64) | 1-D array |
| `std::vector<int>` | list, 1-D `ndarray` (int32) | 1-D array |
| `arma::vec` | list, 1-D or 2-D `ndarray` (float64) | 1-D array |
| `arma::mat` | 2-D `ndarray` (float64, row-major) | 1-D array* |

\* Reshape with `.reshape(rows, cols)` after the call.

2-D numpy arrays passed to `std::vector<double>` or `arma::vec` are
auto-flattened in row-major (C) order.  Fortran-order arrays are made
contiguous first.

---

## Standard headers

The following headers are always available without `#include`:

```
<vector>  <cmath>  <random>  <algorithm>  <numeric>  <stdexcept>  <string>
```

---

## How it works

```
cppFunction(code)
    │
    ├─ parser      find function signatures (return type, name, params)
    ├─ wrapper     generate extern "C" shims + assemble .cpp source
    ├─ compiler    g++ / clang++ / cl.exe → shared library (.so/.dylib/.dll)
    ├─ cache       store in ~/.cache/statscpp/<hash>; skip if already built
    └─ marshal     ctypes bindings + numpy conversion → Python callable
```

Compiled libraries are cached by content hash in `~/.cache/statscpp/`.
The same code string only compiles once, ever, on a given machine.

---

## API

| Function | Description |
|---|---|
| `cppFunction(code)` | Compile C++ function(s), return callable or `{name: callable}` |
| `sourceCpp(path)` | Same, reading from a `.cpp` file |
| `evalCpp(expr)` | Evaluate a single C++ expression, return numpy array |
| `check()` | Print environment diagnostics |

---

## License

GNU GPL v3 — see [LICENSE](LICENSE).
