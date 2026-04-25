"""
statscpp — write C++ statistical code in Python strings, compile and run it.

Quick start
-----------
>>> from statscpp import cppFunction
>>> rnorm = cppFunction('''
...     std::vector<double> rnorm(int n, double mean, double sd) {
...         thread_local std::mt19937_64 gen{std::random_device{}()};
...         std::normal_distribution<double> dist(mean, sd);
...         std::vector<double> out(n);
...         for (auto& v : out) v = dist(gen);
...         return out;
...     }
... ''')
>>> rnorm(10, 0.0, 1.0)

Run ``statscpp.check()`` to verify your compiler and optional Armadillo
installation.
"""
from .api      import cppFunction, sourceCpp, evalCpp, check
from .compiler import CompilationError
from ._version import __version__

__all__ = ["cppFunction", "sourceCpp", "evalCpp", "check", "CompilationError"]
