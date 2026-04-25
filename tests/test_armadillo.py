"""Integration tests for Armadillo support (arma::vec, arma::mat)."""
import numpy as np
import pytest
from statscpp import cppFunction


@pytest.fixture(scope="module")
def normalize_vec():
    return cppFunction("""
        arma::vec normalize(arma::vec x) { return x / arma::norm(x); }
    """)

@pytest.fixture(scope="module")
def mat_inv():
    return cppFunction("""
        arma::mat mat_inv(arma::mat A) { return arma::inv(A); }
    """)

@pytest.fixture(scope="module")
def mat_vec_mul():
    return cppFunction("""
        arma::vec mat_vec_mul(arma::mat A, arma::vec x) { return A * x; }
    """)

@pytest.fixture(scope="module")
def mat_det():
    return cppFunction("""
        double mat_det(arma::mat A) { return arma::det(A); }
    """)


class TestArmaVec:
    def test_normalize(self, normalize_vec):
        result = normalize_vec(np.array([3.0, 4.0]))
        np.testing.assert_allclose(result, [0.6, 0.8])

    def test_returns_ndarray(self, normalize_vec):
        assert isinstance(normalize_vec(np.array([1.0, 0.0])), np.ndarray)


class TestArmaMat:
    def test_inv_is_inverse(self, mat_inv):
        A      = np.array([[1.0, 2.0], [3.0, 4.0]])
        A_inv  = mat_inv(A).reshape(2, 2)
        np.testing.assert_allclose(A_inv @ A, np.eye(2), atol=1e-10)

    def test_det(self, mat_det):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert abs(mat_det(A)[0] - np.linalg.det(A)) < 1e-10

    def test_mat_vec_mul(self, mat_vec_mul):
        A      = np.array([[1.0, 0.0], [0.0, 2.0]])
        x      = np.array([3.0, 4.0])
        result = mat_vec_mul(A, x)
        np.testing.assert_allclose(result, [3.0, 8.0])

    def test_non_square_matrix(self, mat_vec_mul):
        A      = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])   # 2×3
        x      = np.array([1.0, 0.0, 0.0])
        result = mat_vec_mul(A, x)
        np.testing.assert_allclose(result, [1.0, 4.0])


class TestArmaAutoInclude:
    def test_arma_prefix_triggers_include(self):
        fn = cppFunction("""
            arma::vec scale(arma::vec v, double s) { return v * s; }
        """)
        result = fn(np.array([1.0, 2.0, 3.0]), 2.0)
        np.testing.assert_allclose(result, [2.0, 4.0, 6.0])
