import numpy as np

import pxalyze as xa


def test_what_ndarray():
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    result = xa.what(x)

    assert result["type"] is np.ndarray
    assert result["min"] == 0
    assert result["max"] == 5
    assert result["mean"] == np.mean(x)
    assert result["shape"] == (2, 3)
    assert result["dtype"] == np.float64
    assert "keys" not in result
    assert "len" not in result


def test_what_list():
    result = xa.what([1, 2, 3])

    assert result == {"type": list, "len": 3}


def test_what_dict():
    result = xa.what({"a": 0, "b": 1})

    assert result == {"type": dict, "keys": ["a", "b"]}


def test_what_scalar():
    result = xa.what(5)

    assert result == {"type": int}


def test_what_scalar_ndarray():
    x = np.array(3.0)
    result = xa.what(x)

    assert result["min"] == 3.0
    assert result["mean"] == 3.0
    assert result["max"] == 3.0
    assert result["shape"] == ()


def test_rwhat_list_of_arrays():
    items = [np.array(1), np.array(2)]
    result = xa.rwhat(items)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["min"] == 1
    assert result[1]["min"] == 2


def test_rwhat_tuple_of_arrays():
    items = (np.array(1), np.array(2))
    result = xa.rwhat(items)

    assert isinstance(result, list)
    assert len(result) == 2


def test_rwhat_dict_of_arrays():
    items = {"a": np.array(1), "b": np.array(2)}
    result = xa.rwhat(items)

    assert isinstance(result, dict)
    assert result["a"]["min"] == 1
    assert result["b"]["min"] == 2


def test_rwhat_nested():
    items = {"a": [np.array(1), np.array(2)], "b": {"c": np.array(3)}}
    result = xa.rwhat(items)

    assert result["a"][0]["min"] == 1
    assert result["a"][1]["min"] == 2
    assert result["b"]["c"]["min"] == 3


def test_rwhat_leaf_scalar():
    assert xa.rwhat(5) == {"type": int}
