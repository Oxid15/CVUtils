import numpy as np

import pxalyze as xa


class FakeTensor:
    """
    Duck-types the parts of torch.Tensor that tonp
    relies on to avoid bringing torch dependency
    """

    def __init__(self, arr, detached=False, on_cpu=False):
        self._arr = np.asarray(arr)
        self.detached = detached
        self.on_cpu = on_cpu

    def detach(self):
        return FakeTensor(self._arr, detached=True, on_cpu=self.on_cpu)

    def cpu(self):
        assert (
            self.detached
        ), "cpu() called before detach() like real torch.Tensor requires"
        return FakeTensor(self._arr, detached=self.detached, on_cpu=True)

    def __array__(self, dtype=None, copy=None):
        assert self.on_cpu, "tensor was converted to array before being moved to cpu"
        return self._arr


def test_tonp_plain_array():
    x = np.array([1, 2, 3])
    result = xa.tonp(x)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, x)


def test_tonp_list():
    result = xa.tonp([1, 2, 3])

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1, 2, 3])


def test_tonp_fake_tensor_detaches_and_moves_to_cpu():
    arr = np.random.random((4, 3, 8, 8))
    fake = FakeTensor(arr)

    result = xa.tonp(fake)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, arr)


def test_tonp_object_without_detach_or_cpu():
    class NoTensor:
        def __array__(self, dtype=None, copy=None):
            return np.array([1, 2])

    result = xa.tonp(NoTensor())
    np.testing.assert_array_equal(result, [1, 2])


def test_to1_normalizes_to_unit_range():
    x = np.random.random((5, 5)) * 100 + 25

    result = xa.to1(x)

    assert result.min() == 0.0
    assert result.max() == 1.0


def test_to1_constant_array_falls_back_to_zeros(capsys):
    x = np.ones((2, 2))

    result = xa.to1(x)

    np.testing.assert_array_equal(result, np.zeros((2, 2)))
    assert "Failed to normalize" in capsys.readouterr().out


def test_to255_scales_to_0_255():
    x = np.array([1.0, 5.0, 10.0])

    result = xa.to255(x)

    assert result.min() == 0.0
    assert result.max() == 255.0


def test_to255_constant_array_falls_back_to_zeros():
    x = np.full((2, 2), 7.0)

    result = xa.to255(x)

    np.testing.assert_array_equal(result, np.zeros((2, 2)))
