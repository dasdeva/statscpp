import numpy as np
import pytest
from statscpp import cppFunction, evalCpp, CompilationError

# ---------------------------------------------------------------------------
# Helpers — define C++ functions once, reuse across tests
# ---------------------------------------------------------------------------
_RNORM_CODE = """
std::vector<double> rnorm(int n, double mean, double sd) {
    thread_local std::mt19937_64 gen{std::random_device{}()};
    std::normal_distribution<double> dist(mean, sd);
    std::vector<double> out(n);
    for (auto& v : out) v = dist(gen);
    return out;
}
"""

_RUNIF_CODE = """
std::vector<double> runif(int n, double lo, double hi) {
    thread_local std::mt19937_64 gen{std::random_device{}()};
    std::uniform_real_distribution<double> dist(lo, hi);
    std::vector<double> out(n);
    for (auto& v : out) v = dist(gen);
    return out;
}
"""

_DNORM_CODE = """
double dnorm(double x, double mean, double sd) {
    const double PI = std::acos(-1.0);
    double z = (x - mean) / sd;
    return std::exp(-0.5 * z * z) / (sd * std::sqrt(2.0 * PI));
}
"""

_VEC_SCALE_CODE = """
std::vector<double> scale(std::vector<double> x, double factor) {
    for (auto& v : x) v *= factor;
    return x;
}
"""

_MULTI_CODE = """
std::vector<double> add_one(std::vector<double> x) {
    for (auto& v : x) v += 1.0;
    return x;
}

double vec_sum(std::vector<double> x) {
    double s = 0.0;
    for (double v : x) s += v;
    return s;
}
"""


# ---------------------------------------------------------------------------
# cppFunction — basic usage
# ---------------------------------------------------------------------------
def test_rnorm_returns_ndarray():
    rnorm = cppFunction(_RNORM_CODE)
    result = rnorm(10, 0.0, 1.0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (10,)
    assert result.dtype == np.float64


def test_rnorm_mean_approx():
    rnorm = cppFunction(_RNORM_CODE)
    result = rnorm(500, 5.0, 0.1)
    assert abs(np.mean(result) - 5.0) < 0.1


def test_runif_range():
    runif = cppFunction(_RUNIF_CODE)
    result = runif(1000, 2.0, 4.0)
    assert np.all(result >= 2.0) and np.all(result <= 4.0)


def test_scalar_return():
    dnorm = cppFunction(_DNORM_CODE)
    result = dnorm(0.0, 0.0, 1.0)
    assert result.shape == (1,)
    assert abs(result[0] - 0.3989422804014327) < 1e-10


def test_vector_param():
    scale = cppFunction(_VEC_SCALE_CODE)
    result = scale([1.0, 2.0, 3.0], 2.0)
    np.testing.assert_allclose(result, [2.0, 4.0, 6.0])


def test_vector_param_numpy_input():
    scale = cppFunction(_VEC_SCALE_CODE)
    x = np.array([1.0, 2.0, 3.0])
    result = scale(x, 3.0)
    np.testing.assert_allclose(result, [3.0, 6.0, 9.0])


def test_zero_length():
    rnorm = cppFunction(_RNORM_CODE)
    result = rnorm(0, 0.0, 1.0)
    assert result.shape == (0,)


# ---------------------------------------------------------------------------
# Multiple functions in one cppFunction call
# ---------------------------------------------------------------------------
def test_multiple_functions_returns_dict():
    fns = cppFunction(_MULTI_CODE)
    assert isinstance(fns, dict)
    assert set(fns.keys()) == {'add_one', 'vec_sum'}


def test_multiple_functions_correct_results():
    fns = cppFunction(_MULTI_CODE)
    result = fns['add_one']([1.0, 2.0, 3.0])
    np.testing.assert_allclose(result, [2.0, 3.0, 4.0])
    s = fns['vec_sum']([1.0, 2.0, 3.0])
    assert abs(s[0] - 6.0) < 1e-10


# ---------------------------------------------------------------------------
# Caching — same code string must not recompile
# ---------------------------------------------------------------------------
def test_caching():
    fn1 = cppFunction(_RNORM_CODE)
    fn2 = cppFunction(_RNORM_CODE)
    assert fn1.__name__ == fn2.__name__ == "rnorm"


# ---------------------------------------------------------------------------
# evalCpp
# ---------------------------------------------------------------------------
def test_evalcpp_scalar():
    result = evalCpp("std::sqrt(2.0)")
    assert result.shape == (1,)
    assert abs(result[0] - 1.4142135623730951) < 1e-10


def test_evalcpp_expression():
    result = evalCpp("3.14 * 2.0")
    assert abs(result[0] - 6.28) < 1e-10


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
def test_compilation_error():
    with pytest.raises(CompilationError):
        cppFunction("std::vector<double> bad() { return not_defined(); }")


def test_no_functions_raises():
    with pytest.raises(ValueError):
        cppFunction("// just a comment, no function")
