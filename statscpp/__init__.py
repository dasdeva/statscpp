"""
statscpp — write C++ statistical code in Python strings, compile and run it.

What is statscpp?
-----------------
statscpp lets you write C++ code as Python strings, compile it on-the-fly,
and call it like a normal Python function. This is useful for:

  • Running statistical simulations that are performance-critical
  • Using C++ libraries (like Armadillo) from Python
  • Avoiding the overhead of looping in Python

Key benefit: you get C++ speed without writing CMakeLists.txt or build systems.

Quick start (for ML engineers)
------------------------------
Option 1: Use a pre-built function:

  >>> from statscpp.presets import rnorm, mean, variance
  >>> samples = rnorm(10000, mean=5, sd=2)
  >>> print(mean(samples), variance(samples))

Option 2: Write your own C++ function:

  >>> from statscpp import cppFunction
  >>> my_func = cppFunction('''
  ...     double square(double x) {
  ...         return x * x;
  ...     }
  ... ''')
  >>> my_func(5.0)
  25.0

Option 3: Evaluate a quick C++ expression:

  >>> from statscpp import evalCpp
  >>> evalCpp("2.0 + 3.0")  # returns np.array([5.0])

What happens behind the scenes?
--------------------------------
  1. Your C++ code is analyzed to find function signatures
  2. Automatic wrapper code is generated to interface with Python
  3. The C++ compiler on your system compiles it to a shared library
  4. Results are cached so recompiling doesn't happen again
  5. ctypes calls the compiled code and converts results to NumPy arrays

Requirements
-----------
  • A C++ compiler: g++, clang++, or MSVC
    macOS:   xcode-select --install
    Ubuntu:  sudo apt install build-essential
    Windows: Visual Studio Build Tools or MinGW

Run ``statscpp.check()`` to verify your setup.
"""
from .api      import cppFunction, sourceCpp, evalCpp, check
from .compiler import CompilationError
from . import presets
from ._version import __version__

__all__ = [
    "cppFunction",
    "sourceCpp",
    "evalCpp",
    "check",
    "CompilationError",
    "presets",  # Re-export the presets module
]
