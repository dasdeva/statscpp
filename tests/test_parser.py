"""Unit tests for statscpp.parser — C++ signature parsing."""
import pytest
from statscpp.parser import parse_params, find_functions
from statscpp.types  import INT, DOUBLE, VEC, IVEC, AVEC, AMAT


class TestParseParams:
    def test_empty(self):
        assert parse_params("") == []

    def test_single_int(self):
        assert parse_params("int n") == [(INT, "n")]

    def test_single_double(self):
        assert parse_params("double x") == [(DOUBLE, "x")]

    def test_vec_param(self):
        assert parse_params("std::vector<double> v") == [(VEC, "v")]

    def test_ivec_param(self):
        assert parse_params("std::vector<int> idx") == [(IVEC, "idx")]

    def test_arma_vec(self):
        assert parse_params("arma::vec x") == [(AVEC, "x")]

    def test_arma_mat(self):
        assert parse_params("arma::mat A") == [(AMAT, "A")]

    def test_multiple_params(self):
        result = parse_params("int n, double mu, double sigma")
        assert result == [(INT, "n"), (DOUBLE, "mu"), (DOUBLE, "sigma")]

    def test_default_value_stripped(self):
        result = parse_params("int n, double mu = 0.0")
        assert result == [(INT, "n"), (DOUBLE, "mu")]

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError):
            parse_params("std::string s")


class TestFindFunctions:
    def test_single_function(self):
        code = "double sq(double x) { return x * x; }"
        fns  = find_functions(code)
        assert len(fns) == 1
        ret, name, params = fns[0]
        assert ret    == DOUBLE
        assert name   == "sq"
        assert params == [(DOUBLE, "x")]

    def test_multiple_functions(self):
        code = """
            double sq(double x) { return x * x; }
            int sign(double x) { return x > 0 ? 1 : -1; }
        """
        fns = find_functions(code)
        assert [n for _, n, _ in fns] == ["sq", "sign"]

    def test_vector_return(self):
        code = "std::vector<double> zeros(int n) { return std::vector<double>(n, 0.0); }"
        fns  = find_functions(code)
        assert fns[0][0] == VEC

    def test_unsupported_type_skipped(self):
        code = """
            std::string bad(std::string s) { return s; }
            double good(double x) { return x; }
        """
        fns = find_functions(code)
        assert len(fns) == 1
        assert fns[0][1] == "good"

    def test_no_functions_returns_empty(self):
        assert find_functions("// just a comment") == []
