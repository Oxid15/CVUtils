import pxalyze as xa


def test_lm_single_function_on_scalar():
    assert xa.lm(5, lambda x: x + 1) == 6


def test_lm_multiple_functions_on_scalar():
    assert xa.lm(5, lambda x: x + 1, lambda x: x * 2) == 12


def test_lm_single_function_on_list():
    assert xa.lm([1, 2], lambda x: x + 1) == [2, 3]


def test_lm_multiple_functions_on_list():
    assert xa.lm([1, 2], lambda x: x + 1, lambda x: x + 1) == [3, 4]


def test_lm_tuple_input_maps_like_a_list():
    assert xa.lm((1, 2), lambda x: x + 1) == [2, 3]


def test_lm_no_functions_returns_input_unchanged():
    assert xa.lm(5) == 5
    assert xa.lm([1, 2]) == [1, 2]


def test_lm_chains_in_declared_order():
    # order matters: (x * 2) + 1, not (x + 1) * 2
    assert xa.lm(3, lambda x: x * 2, lambda x: x + 1) == 7
