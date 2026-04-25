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
# std::vector<int>
# ---------------------------------------------------------------------------
_IVEC_CODE = """
int ivec_sum(std::vector<int> x) {
    int s = 0;
    for (int v : x) s += v;
    return s;
}
"""

def test_ivec_from_list():
    fn = cppFunction(_IVEC_CODE)
    result = fn([1, 2, 3, 4])
    assert result[0] == 10.0

def test_ivec_from_numpy():
    fn = cppFunction(_IVEC_CODE)
    import numpy as np
    result = fn(np.array([5, 6, 7], dtype=np.int32))
    assert result[0] == 18.0


# ---------------------------------------------------------------------------
# 2D numpy array (matrix) input
# ---------------------------------------------------------------------------
_TRACE_CODE = """
double trace(std::vector<double> mat, int rows, int cols) {
    double s = 0.0;
    int n = std::min(rows, cols);
    for (int i = 0; i < n; i++) s += mat[i * cols + i];
    return s;
}
"""

_MATMUL_CODE = """
std::vector<double> matmul(
    std::vector<double> A, int Ar, int Ac,
    std::vector<double> B, int Br, int Bc
) {
    if (Ac != Br) throw std::invalid_argument("inner dims mismatch");
    std::vector<double> C(Ar * Bc, 0.0);
    for (int i = 0; i < Ar; i++)
        for (int k = 0; k < Ac; k++)
            for (int j = 0; j < Bc; j++)
                C[i * Bc + j] += A[i * Ac + k] * B[k * Bc + j];
    return C;
}
"""

def test_matrix_trace():
    fn = cppFunction(_TRACE_CODE)
    import numpy as np
    A = np.eye(3)
    result = fn(A, *A.shape)   # 2D array auto-flattened, shape unpacked as rows, cols
    assert abs(result[0] - 3.0) < 1e-10

def test_matrix_trace_nonsquare():
    fn = cppFunction(_TRACE_CODE)
    import numpy as np
    A = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])   # 2x3
    result = fn(A, *A.shape)
    assert abs(result[0] - 6.0) < 1e-10   # 1 + 5

def test_matmul():
    fn = cppFunction(_MATMUL_CODE)
    import numpy as np
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    result = fn(A, *A.shape, B, *B.shape).reshape(2, 2)
    expected = A @ B
    np.testing.assert_allclose(result, expected)

def test_matrix_row_major_order():
    # Verify that C-order (row-major) flattening is used
    fn = cppFunction(_TRACE_CODE)
    import numpy as np
    # Fortran-order array should still give the right trace after auto-flatten
    A = np.asfortranarray(np.eye(4))
    result = fn(A, *A.shape)
    assert abs(result[0] - 4.0) < 1e-10


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
def test_compilation_error():
    with pytest.raises(CompilationError):
        cppFunction("std::vector<double> bad() { return not_defined(); }")


def test_no_functions_raises():
    with pytest.raises(ValueError):
        cppFunction("// just a comment, no function")
