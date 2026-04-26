# Getting Started with statscpp

If you're new to this codebase, start here. This guide shows you how to use statscpp and explains just enough about the implementation to understand what's happening.

## Installation & Setup

1. **Install a C++ compiler** (if you don't have one):
   ```bash
   # macOS
   xcode-select --install
   
   # Ubuntu/Debian
   sudo apt install build-essential
   
   # Fedora
   sudo dnf install gcc-c++
   ```

2. **Verify your setup:**
   ```python
   from statscpp import check
   check()  # Prints "✓ Ready to use!" if everything is set up
   ```

3. **Install statscpp:**
   ```bash
   pip install -e .  # From the repo directory
   ```

## Three Ways to Use statscpp

### Option 1: Use Pre-built Functions (Easiest)

If you just need random numbers or basic statistics, use the presets:

```python
from statscpp.presets import rnorm, mean, variance

# Generate 10,000 normally-distributed random numbers
samples = rnorm(10000, mean=5, sd=2)

# Compute statistics (much faster than NumPy for huge arrays)
mu = mean(samples)
sigma_sq = variance(samples)
```

**When to use this:** You want fast random sampling or statistics without writing C++.

---

### Option 2: Write Your Own C++ Function (Most Common)

For custom logic, write a C++ function as a Python string:

```python
from statscpp import cppFunction
import numpy as np

# Define a function in C++
my_code = """
    // Compute the sum of squared deviations from the mean
    double sum_sq_dev(std::vector<double> x) {
        double m = 0;
        for (auto v : x) m += v;
        m /= x.size();
        
        double ss = 0;
        for (auto v : x) {
            double d = v - m;
            ss += d * d;
        }
        return ss;
    }
"""

# Compile it
sum_sq_dev = cppFunction(my_code)

# Use it like a normal Python function
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
result = sum_sq_dev(data)
print(result)  # 10.0
```

**When to use this:** Your calculation is performance-critical or uses C++ features.

**Tips:**
- Start simple: get it working in Python first, then convert to C++
- Standard C++ libraries are automatically included (#include <vector>, <cmath>, etc.)
- You can use #include <armadillo> for matrix operations (need to install libarmadillo)
- Exceptions are caught and converted to Python exceptions

---

### Option 3: Evaluate Quick C++ Expressions (For Testing)

For one-off calculations:

```python
from statscpp import evalCpp

result = evalCpp("2.0 + 3.0")
# → np.array([5.0])

result = evalCpp("std::sqrt(16.0)")
# → np.array([4.0])
```

**When to use this:** Testing or quick calculations. Not recommended for production code.

---

## What Types Can You Use?

Your C++ functions can accept and return:

| C++ Type | Python Equivalent | Notes |
|----------|---|---|
| `int` | Python `int` | Also accepts `long`, `size_t` |
| `double` | Python `float` | Also accepts `float` |
| `std::vector<double>` | NumPy `np.ndarray` (1-D, float64) | 1-D arrays |
| `std::vector<int>` | NumPy `np.ndarray` (1-D, int32) | Integer arrays |
| `arma::vec` | NumPy `np.ndarray` (1-D, float64) | Armadillo vectors (faster) |
| `arma::mat` | NumPy `np.ndarray` (2-D, float64) | Armadillo matrices (faster) |

**Unsupported:** `std::string`, custom classes, pointers, etc. (they're silently skipped or cause errors)

---

## Common Patterns

### Pattern 1: Vectorized Computation

```python
from statscpp import cppFunction

square_all = cppFunction("""
    std::vector<double> square_all(std::vector<double> x) {
        std::vector<double> out(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            out[i] = x[i] * x[i];
        }
        return out;
    }
""")

import numpy as np
result = square_all(np.array([1, 2, 3, 4, 5]))
# → [1, 4, 9, 16, 25]
```

### Pattern 2: Reduction (Fold/Aggregate)

```python
from statscpp import cppFunction

total = cppFunction("""
    double total(std::vector<double> x) {
        double sum = 0;
        for (auto v : x) sum += v;
        return sum;
    }
""")

result = total(np.array([1, 2, 3, 4, 5]))
# → 15.0
```

### Pattern 3: Multiple Functions

You can define multiple functions in one string:

```python
from statscpp import cppFunction

funcs = cppFunction("""
    double add(double a, double b) {
        return a + b;
    }
    
    double multiply(double a, double b) {
        return a * b;
    }
""")

# When multiple functions exist, you get a dict
print(funcs['add'](2, 3))  # 5
print(funcs['multiply'](2, 3))  # 6
```

---

## Understanding What Happens Behind the Scenes

When you call `cppFunction(code)`:

1. **Parse:** Extract function signatures (names, parameter types, return types)
2. **Generate:** Create C "shim" functions that bridge Python and C++
3. **Compile:** Use your system's C++ compiler to create a shared library
4. **Cache:** Store the compiled library so recompiling doesn't happen twice
5. **Wrap:** Return a Python function that calls the compiled code

All steps are automatic. You just write the C++ code and use it.

**One important detail:** The first call to a new function takes 1-5 seconds (compilation). Subsequent calls are instant (uses cached library).

---

## Debugging: What to Do When Things Break

### Compiler Not Found

```
❌ No C++ compiler found on PATH.
```

**Fix:** Install a compiler (see "Installation & Setup" above).

### Syntax Error in C++

```
❌ C++ compiler error. Your code has a syntax or type error:

error: 'std::vector<std::string>' is not supported
```

**Fix:** Check your C++ syntax. Use only supported types (see "What Types Can You Use?").

### Type Error

```
❌ Unsupported return type or parameter type: 'MyClass'

Supported types:
  • int, long, size_t → Python int
  • double, float → Python float
  ...
```

**Fix:** Use only supported types, or create a simpler function that returns a supported type.

---

## File Structure

```
statscpp/
├── __init__.py          ← Main exports (cppFunction, evalCpp, sourceCpp)
├── api.py               ← User-facing functions
├── presets.py           ← Pre-compiled common functions (rnorm, mean, etc.)
├── parser.py            ← Extract C++ signatures
├── types.py             ← Type system
├── wrapper.py           ← Generate shim code
├── compiler.py          ← Detect and invoke C++ compiler
├── cache.py             ← Cache compiled libraries
├── marshal.py           ← Python ↔ C data conversion
├── armadillo.py         ← Armadillo support
└── include/
    └── statscpp.hpp     ← C++ headers
```

**Key insight:** Most users only interact with `api.py` and `presets.py`. The other modules handle the compilation machinery.

---

## Next Steps

- **For quick wins:** Use `presets.py` functions
- **For custom logic:** Learn the `cppFunction()` pattern
- **For advanced topics:** Read [PIPELINE.md](PIPELINE.md)
- **For library code:** Read the docstrings in each module (they explain the "why")

---

## Examples

### Example 1: Fast Sampling

```python
from statscpp.presets import rnorm
import time

# NumPy sampling
import numpy as np
start = time.time()
np_samples = np.random.normal(loc=0, scale=1, size=1_000_000)
print(f"NumPy: {time.time() - start:.3f}s")

# statscpp sampling (faster)
start = time.time()
scpp_samples = rnorm(1_000_000, mean=0, sd=1)
print(f"statscpp: {time.time() - start:.3f}s")
```

### Example 2: Simulation (Monte Carlo)

```python
from statscpp import cppFunction

pi_estimate = cppFunction("""
    double estimate_pi(int n_samples) {
        thread_local std::mt19937_64 gen{std::random_device{}()};
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        int inside = 0;
        for (int i = 0; i < n_samples; i++) {
            double x = dist(gen);
            double y = dist(gen);
            if (x*x + y*y <= 1.0) inside++;
        }
        
        return 4.0 * inside / n_samples;
    }
""")

# Estimate π using 10 million samples
result = pi_estimate(10_000_000)
print(result)  # ≈ 3.14159...
```

---

## Questions?

- **How fast is statscpp?** As fast as compiled C++. For loops and math, expect 10-100× faster than NumPy.
- **Can I use NumPy inside my C++ code?** No. You're writing C++, not Python. But you can use Armadillo for linear algebra.
- **What about GPU acceleration?** Not yet. statscpp compiles to CPU code. GPU support could be added later.
- **Can I use external libraries?** Yes, if you can link against them (advanced use case). See `armadillo.py` for an example.
