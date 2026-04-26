# How statscpp Works: The Compilation Pipeline

> 💡 A mental model for understanding the codebase

## The Big Picture

You write a Python string containing C++ code. statscpp turns it into a working Python function. Here's what happens under the hood:

```
Your C++ code (string)
    ↓
[Parser] → Extract function signatures
    ↓
[Wrapper] → Generate "shim" functions that bridge Python and C++
    ↓
[Compiler] → Compile to a shared library (.so / .dll / .dylib)
    ↓
[Cache] → Store it so we don't recompile next time
    ↓
[Marshal] → Convert Python args to C, call the shim, convert results back
    ↓
Your callable Python function
```

## Each Step Explained

### 1. **Parser** (`parser.py`)

**What it does:** Reads your C++ code and extracts function signatures.

**Why:** Python's ctypes can only call extern "C" functions. But your code might have C++ features (templates, vectors). So we need to know:
- What function names exist?
- What do they return?
- What are their parameters?

**Example:**
```python
code = """
    std::vector<double> rnorm(int n, double mean) { ... }
"""
# Parser extracts: ("vec", "rnorm", [("int", "n"), ("double", "mean")])
```

### 2. **Types System** (`types.py`)

**What it does:** Maps C++ types to "canonical tags" used throughout the pipeline.

**Why:** Different modules need to talk about types. Instead of each module parsing types independently, we use universal tags:
- `"int"` → int, long, size_t
- `"double"` → double, float
- `"vec"` → std::vector<double>
- `"avec"` → arma::vec (Armadillo)
- `"amat"` → arma::mat

This lets each module stay focused and simple.

### 3. **Wrapper** (`wrapper.py`)

**What it does:** Generates the glue code that connects Python to C++.

**Why:** Python's ctypes can only call extern "C" functions with flat C-style arguments. But we want users to write normal C++ code with vectors, exceptions, etc.

The solution: Generate a shim function for each user function.

**Example:**

User writes:
```cpp
std::vector<double> rnorm(int n) { ... }
```

We generate:
```cpp
extern "C" int __shim_rnorm(double* __out, int* __n_out, int n) {
    try {
        std::vector<double> result = rnorm(n);
        *__n_out = result.size();
        for (int i=0; i < *__n_out; i++) __out[i] = result[i];
        return 0;  // success
    } catch (...) {
        *__n_out = 0;
        return 1;  // error
    }
}
```

The shim:
1. Accepts flattened C arguments
2. Reconstructs high-level types
3. Calls your function
4. Flattens results back to C arguments
5. Handles errors

### 4. **Compiler** (`compiler.py`)

**What it does:** Finds your C++ compiler and runs it.

**Why:** To turn C++ code into machine code.

**Process:**
```
1. Detect compiler: g++, clang++, or MSVC?
2. Set compiler flags: -O2 (optimize), -std=c++17, -shared (make a library)
3. Compile: compiler.exe source.cpp -o library.so
```

### 5. **Cache** (`cache.py`)

**What it does:** Stores compiled libraries so recompilation is instant.

**Why:** Compilation is slow (~1-5 seconds). If the same code appears again, just reuse the old library.

**How:**
```
1. Hash the code: md5(version + code) → unique key
2. Check ~/.cache/statscpp/ for key.so
3. If exists: load it (instant)
4. If not: compile it, save to cache

Two-level cache:
  - Disk: Survives between Python sessions
  - In-process: Fast lookups within a session
```

### 6. **Marshal** (`marshal.py`)

**What it does:** Converts Python values to C and back.

**Why:** Python and C have different memory layouts and types.

**Example:**
```python
# User calls:
result = rnorm(n=10, mean=0.0)

# Internally:
- Convert 10 (Python int) → ctypes.c_int(10)
- Convert 0.0 (Python float) → ctypes.c_double(0.0)
- Call the C shim with these values
- Extract results from the C output buffer
- Convert back to np.ndarray
```

**Memory safety:** For arrays, we keep references to NumPy arrays until after the C call. This prevents Python from freeing the memory while C is still using it.

## The Entry Points

Users interact via three functions in `api.py`:

### `cppFunction(code: str) → Callable`
Compile one or more C++ functions and return callable(s).

Pipeline:
```
code → Parser → Wrapper.build_source() → Compiler → Cache.load() → Marshal.make_callable() → Callable
```

### `evalCpp(expr: str) → np.ndarray`
Evaluate a single C++ expression that returns a number or vector.

Pipeline:
```
"2 + 3" → Wrapper (special case) → Compiler → Cache.load() → Marshal → np.array([5.0])
```

### `sourceCpp(path: str) → Callable`
Load C++ code from a file and compile it.

Pipeline:
```
file.cpp → read text → cppFunction()
```

## Why This Design?

**Separation of concerns:** Each module has one job:
- Parser: Parse C++ signatures ✓
- Types: Define the type system ✓
- Wrapper: Generate glue code ✓
- Compiler: Find compiler and compile ✓
- Cache: Store and load libraries ✓
- Marshal: Convert between Python and C ✓

**Testability:** Each module can be tested independently.

**Maintainability:** To add support for a new type (e.g., std::complex), you only change `types.py`, `wrapper.py`, and `marshal.py`.

**Performance:** The cache means users only pay the compilation cost once per unique code string.

## For ML Engineers: The Simple Path

You don't need to understand all this. Just use the presets:

```python
from statscpp.presets import rnorm, mean, variance

samples = rnorm(10000, mean=5, sd=2)  # Fast random sampling
print(mean(samples))  # Fast computation
```

If you need custom C++, follow this pattern:

```python
from statscpp import cppFunction

my_func = cppFunction("""
    double my_function(double x) {
        return x * x;  // Your logic here
    }
""")

result = my_func(5.0)
```

And if something breaks, the error messages will tell you exactly what's wrong and how to fix it.

## Additional Resources

- **`compiler.py`**: How to detect and invoke your system's C++ compiler
- **`cache.py`**: How to avoid recompiling the same code
- **`marshal.py`**: How Python ↔ C data conversion works
- **`types.py`**: Supported C++ types and how they map to Python/NumPy
