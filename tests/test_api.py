"""Integration tests for the full statscpp pipeline (no Armadillo)."""
import numpy as np
import pytest
from statscpp import cppFunction, evalCpp, CompilationError


# ---------------------------------------------------------------------------
# Fixtures — compiled functions reused across tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rnorm():
    return cppFunction("""
        std::vector<double> rnorm(int n, double mean, double sd) {
            thread_local std::mt19937_64 gen{std::random_device{}()};
            std::normal_distribution<double> dist(mean, sd);
            std::vector<double> out(n);
            for (auto& v : out) v = dist(gen);
            return out;
        }
    """)

@pytest.fixture(scope="module")
def runif():
    return cppFunction("""
        std::vector<double> runif(int n, double lo, double hi) {
            thread_local std::mt19937_64 gen{std::random_device{}()};
            std::uniform_real_distribution<double> dist(lo, hi);
            std::vector<double> out(n);
            for (auto& v : out) v = dist(gen);
            return out;
        }
    """)

@pytest.fixture(scope="module")
def dnorm():
    return cppFunction("""
        double dnorm(double x, double mean, double sd) {
            const double PI = std::acos(-1.0);
            double z = (x - mean) / sd;
            return std::exp(-0.5 * z * z) / (sd * std::sqrt(2.0 * PI));
        }
    """)

@pytest.fixture(scope="module")
def scale():
    return cppFunction("""
        std::vector<double> scale(std::vector<double> x, double f) {
            for (auto& v : x) v *= f;
            return x;
        }
    """)

@pytest.fixture(scope="module")
def ivec_sum():
    return cppFunction("""
        int ivec_sum(std::vector<int> x) {
            int s = 0;
            for (int v : x) s += v;
            return s;
        }
    """)

@pytest.fixture(scope="module")
def trace():
    return cppFunction("""
        double trace(std::vector<double> mat, int rows, int cols) {
            double s = 0.0;
            int n = std::min(rows, cols);
            for (int i = 0; i < n; i++) s += mat[i * cols + i];
            return s;
        }
    """)


# ---------------------------------------------------------------------------
# cppFunction — return type and shape
# ---------------------------------------------------------------------------
class TestReturnTypes:
    def test_ndarray(self, rnorm):
        assert isinstance(rnorm(10, 0.0, 1.0), np.ndarray)

    def test_dtype_float64(self, rnorm):
        assert rnorm(10, 0.0, 1.0).dtype == np.float64

    def test_shape(self, rnorm):
        assert rnorm(10, 0.0, 1.0).shape == (10,)

    def test_scalar_return_is_1d(self, dnorm):
        result = dnorm(0.0, 0.0, 1.0)
        assert result.shape == (1,)

    def test_scalar_value(self, dnorm):
        assert abs(dnorm(0.0, 0.0, 1.0)[0] - 0.3989422804014327) < 1e-10

    def test_zero_length(self, rnorm):
        assert rnorm(0, 0.0, 1.0).shape == (0,)


# ---------------------------------------------------------------------------
# cppFunction — statistical correctness
# ---------------------------------------------------------------------------
class TestStatistical:
    def test_rnorm_mean(self, rnorm):
        r = rnorm(500, 5.0, 0.1)
        assert abs(np.mean(r) - 5.0) < 0.1

    def test_runif_range(self, runif):
        r = runif(1000, 2.0, 4.0)
        assert np.all(r >= 2.0) and np.all(r <= 4.0)


# ---------------------------------------------------------------------------
# cppFunction — input types
# ---------------------------------------------------------------------------
class TestInputTypes:
    def test_vec_from_list(self, scale):
        np.testing.assert_allclose(scale([1.0, 2.0, 3.0], 2.0), [2.0, 4.0, 6.0])

    def test_vec_from_numpy(self, scale):
        np.testing.assert_allclose(scale(np.array([1.0, 2.0]), 3.0), [3.0, 6.0])

    def test_2d_array_flattened_rowmajor(self, trace):
        A = np.eye(3)
        assert abs(trace(A, *A.shape)[0] - 3.0) < 1e-10

    def test_2d_nonsquare(self, trace):
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert abs(trace(A, *A.shape)[0] - 6.0) < 1e-10   # 1 + 5

    def test_fortran_order_auto_converted(self, trace):
        A = np.asfortranarray(np.eye(4))
        assert abs(trace(A, *A.shape)[0] - 4.0) < 1e-10

    def test_ivec_from_list(self, ivec_sum):
        assert ivec_sum([1, 2, 3, 4])[0] == 10.0

    def test_ivec_from_numpy(self, ivec_sum):
        assert ivec_sum(np.array([5, 6, 7], dtype=np.int32))[0] == 18.0


# ---------------------------------------------------------------------------
# cppFunction — multiple functions
# ---------------------------------------------------------------------------
class TestMultipleFunctions:
    @pytest.fixture(scope="class")
    def fns(self):
        return cppFunction("""
            std::vector<double> add_one(std::vector<double> x) {
                for (auto& v : x) v += 1.0;
                return x;
            }
            double vec_sum(std::vector<double> x) {
                double s = 0.0;
                for (double v : x) s += v;
                return s;
            }
        """)

    def test_returns_dict(self, fns):
        assert isinstance(fns, dict)
        assert set(fns.keys()) == {"add_one", "vec_sum"}

    def test_add_one(self, fns):
        np.testing.assert_allclose(fns["add_one"]([1.0, 2.0, 3.0]), [2.0, 3.0, 4.0])

    def test_vec_sum(self, fns):
        assert abs(fns["vec_sum"]([1.0, 2.0, 3.0])[0] - 6.0) < 1e-10


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def test_caching_same_code_same_name():
    code = "double identity(double x) { return x; }"
    f1 = cppFunction(code)
    f2 = cppFunction(code)
    assert f1.__name__ == f2.__name__ == "identity"


# ---------------------------------------------------------------------------
# evalCpp
# ---------------------------------------------------------------------------
class TestEvalCpp:
    def test_sqrt(self):
        r = evalCpp("std::sqrt(2.0)")
        assert r.shape == (1,)
        assert abs(r[0] - 1.4142135623730951) < 1e-10

    def test_arithmetic(self):
        assert abs(evalCpp("3.14 * 2.0")[0] - 6.28) < 1e-10


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestErrors:
    def test_compilation_error(self):
        with pytest.raises(CompilationError):
            cppFunction("double bad() { return not_defined(); }")

    def test_no_functions_raises_value_error(self):
        with pytest.raises(ValueError):
            cppFunction("// just a comment")
