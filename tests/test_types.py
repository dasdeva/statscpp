"""Unit tests for statscpp.types — type normalization."""
import pytest
from statscpp.types import normalize, VEC, IVEC, AVEC, AMAT, DOUBLE, INT


@pytest.mark.parametrize("raw,expected", [
    ("int",                      INT),
    ("long",                     INT),
    ("size_t",                   INT),
    ("double",                   DOUBLE),
    ("float",                    DOUBLE),
    ("std::vector<double>",      VEC),
    ("vector<double>",           VEC),
    ("std::vector< double >",    VEC),
    ("std::vector<int>",         IVEC),
    ("vector<int>",              IVEC),
    ("arma::vec",                AVEC),
    ("arma::mat",                AMAT),
])
def test_normalize_valid(raw, expected):
    assert normalize(raw) == expected


def test_normalize_unsupported_raises():
    with pytest.raises(ValueError, match="Unsupported type"):
        normalize("std::string")


def test_normalize_strips_whitespace():
    assert normalize("  double  ") == DOUBLE
