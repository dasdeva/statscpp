"""
Pre-compiled common statistical functions.

For ML engineers new to C++, these are ready-to-use functions without
needing to write C++ code. They're faster than NumPy for certain operations.

Example
-------
>>> from statscpp.presets import rnorm, mean, variance
>>> samples = rnorm(1000, mean=0, sd=1)
>>> print(mean(samples))
"""

from .api import cppFunction
import numpy as np

# ============================================================================
# Random number generation (faster than NumPy for large batches)
# ============================================================================

rnorm = cppFunction("""
    std::vector<double> rnorm(int n, double mean, double sd) {
        if (n < 0) throw std::invalid_argument("n must be >= 0");
        if (sd <= 0) throw std::invalid_argument("sd must be > 0");
        thread_local std::mt19937_64 gen{std::random_device{}()};
        std::normal_distribution<double> dist(mean, sd);
        std::vector<double> out(n);
        for (auto& x : out) x = dist(gen);
        return out;
    }
""")
rnorm.__doc__ = """Generate n standard-normal random variates.

Parameters
----------
n : int
    Number of samples
mean : float
    Mean (default: 0)
sd : float
    Standard deviation (default: 1)

Returns
-------
array of shape (n,)
    Random samples from Normal(mean, sd)
"""

runif = cppFunction("""
    std::vector<double> runif(int n, double min, double max) {
        if (n < 0) throw std::invalid_argument("n must be >= 0");
        if (min >= max) throw std::invalid_argument("min must be < max");
        thread_local std::mt19937_64 gen{std::random_device{}()};
        std::uniform_real_distribution<double> dist(min, max);
        std::vector<double> out(n);
        for (auto& x : out) x = dist(gen);
        return out;
    }
""")
runif.__doc__ = """Generate n uniform random variates.

Parameters
----------
n : int
    Number of samples
min : float
    Minimum (inclusive)
max : float
    Maximum (exclusive)

Returns
-------
array of shape (n,)
    Random samples from Uniform(min, max)
"""

# ============================================================================
# Descriptive statistics (faster than NumPy for huge arrays)
# ============================================================================

mean = cppFunction("""
    double mean(std::vector<double> x) {
        if (x.empty()) throw std::invalid_argument("input cannot be empty");
        double sum = 0;
        for (auto v : x) sum += v;
        return sum / x.size();
    }
""")
mean.__doc__ = """Compute the mean of an array."""

variance = cppFunction("""
    double variance(std::vector<double> x) {
        if (x.size() < 2) throw std::invalid_argument("need at least 2 elements");
        double m = 0;
        for (auto v : x) m += v;
        m /= x.size();
        
        double ss = 0;
        for (auto v : x) {
            double d = v - m;
            ss += d * d;
        }
        return ss / (x.size() - 1);  // sample variance
    }
""")
variance.__doc__ = """Compute the sample variance of an array."""

std = cppFunction("""
    double std(std::vector<double> x) {
        if (x.size() < 2) throw std::invalid_argument("need at least 2 elements");
        double m = 0;
        for (auto v : x) m += v;
        m /= x.size();
        
        double ss = 0;
        for (auto v : x) {
            double d = v - m;
            ss += d * d;
        }
        return std::sqrt(ss / (x.size() - 1));
    }
""")
std.__doc__ = """Compute the sample standard deviation of an array."""

# ============================================================================
# Probability functions (more accurate than NumPy approximations)
# ============================================================================

dnorm = cppFunction("""
    std::vector<double> dnorm(std::vector<double> x, double mean, double sd) {
        if (sd <= 0) throw std::invalid_argument("sd must be > 0");
        std::vector<double> out(x.size());
        constexpr double pi = 3.14159265358979323846;
        for (size_t i = 0; i < x.size(); i++) {
            double z = (x[i] - mean) / sd;
            out[i] = std::exp(-0.5 * z * z) / (sd * std::sqrt(2.0 * pi));
        }
        return out;
    }
""")
dnorm.__doc__ = """Compute normal PDF at given points."""

pnorm = cppFunction("""
    std::vector<double> pnorm(std::vector<double> x, double mean, double sd) {
        if (sd <= 0) throw std::invalid_argument("sd must be > 0");
        std::vector<double> out(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            out[i] = 0.5 * std::erfc(-(x[i] - mean) / (sd * std::sqrt(2.0)));
        }
        return out;
    }
""")
pnorm.__doc__ = """Compute normal CDF at given points."""
