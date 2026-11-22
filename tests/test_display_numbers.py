"""Displayed measurements carry their uncertainty's decimal place through reports and tables."""

import pytest

from src.gimap.shared.numbers import format_number


@pytest.mark.parametrize("value,error,plain,compact", [
    (8.93351, 0.00219, "8.9335 ± 0.0022", "8.9335(22)"),
    (0.670564, 0.0247, "0.671 ± 0.025", "0.671(25)"),
    (1009.28, 0.639, "1009.28 ± 0.64", "1009.28(64)"),
    (0.50124, 0.0119, "0.501 ± 0.012", "0.501(12)"),
    (-1.305, 0.1, "-1.30 ± 0.10", "-1.30(10)"),
    (1.2674, 0.0999, "1.27 ± 0.10", "1.27(10)"),
    (-0.003, 0.66, "0.00 ± 0.66", "0.00(66)"),
    (0, 0.00203, "0.0000 ± 0.0020", "0.0000(20)"),
    (12345, 2190, "1.23e+4 ± 2.2e+3", "1.23(22)e+4"),
    (-7.386e-25, 1.23e-26, "-7.39e-25 ± 1.2e-26", "-7.39(12)e-25"),
    (8.735e250, 6.777e248, "8.735e+250 ± 6.8e+248", "8.735(68)e+250"),
    (0, 2.19e-310, "0.0e-310 ± 2.2e-310", "0.0(22)e-310"),
])
def test_round_pair(value, error, plain, compact):
    assert format_number(value, error) == plain
    assert format_number(value, error, compact=True) == compact
    assert format_number(value, error, value_only=True) == plain.split(" ± ")[0]
    assert format_number(error, error, value_only=True) == plain.split(" ± ")[1]


def test_zero_uncertainty_and_unavailable_measurements():
    assert format_number(1.23456, 0) == "1.23 ± 0"
    assert format_number(2.675) == "2.68"
    assert format_number(10**350) == "1e+350"
    assert format_number(1.23456e-8, 0, compact=True) == "1.23(0)e-08"
    for error in (None, -1, float("nan"), float("inf")):
        assert format_number(1.23456, error, digits=4) == "1.235"
    for value in (None, float("nan"), float("-inf")):
        assert format_number(value, 1) == "—"
