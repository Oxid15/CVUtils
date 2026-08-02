import os

import numpy as np
import pytest

import pxalyze as xa

from tests.conftest import read_written_image

cv2 = pytest.importorskip("cv2")


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize(
    "shape",
    [
        (256, 256),  # HW, single image
        (256, 256, 3),  # HWC, single image
        (3, 256, 256),  # CHW, single image
        (256, 256, 1),  # HWC, single channel
        (1, 256, 256),  # CHW, single channel
        (1, 3, 256, 256),  # BCHW, batch of 1
        (10, 3, 256, 256),  # BCHW, batch of 10
        (10, 256, 256, 3),  # BHWC, batch of 10
    ],
)
def test_atest_handles_a_variety_of_shapes(shape):
    x = np.random.random(shape)

    assert xa.atest(x, post="shape") is True

    img = read_written_image("shape")
    assert img.ndim == 3
    assert img.shape[2] == 3  # cv2 always reads BGR


def test_atest_normalizes_out_of_range_values():
    x = np.random.random((256, 256, 3)) * 10000

    assert xa.atest(x, post="big") is True

    img = read_written_image("big")
    assert img.min() >= 0
    assert img.max() <= 255
    # a wide-range input should use a meaningful chunk of the output range
    assert img.max() - img.min() > 50


def test_atest_unit_range_no_norm():
    x = np.zeros((4, 4, 3))
    x[0, 0, 0] = 1.0  # already within [0, 1], should just be scaled by 255

    assert xa.atest(x, post="unit") is True

    img = read_written_image("unit")
    assert img.max() == 255
    assert img.min() == 0


def test_atest_batch_tiles_and_labels_each_tile():
    x = np.random.random((43, 3, 32, 32))

    assert xa.atest(x, post="batch") is True

    img = read_written_image("batch")
    h_count, w_count = xa._calculate_shape(x)

    # labeled rows add 16px of header per tile row on top of the raw tile height
    assert img.shape[0] == 32 * h_count + h_count * 16
    assert img.shape[1] == 32 * w_count


@pytest.mark.parametrize("batch", [1, 2, 5, 7, 11, 20])
def test_atest_arbitrary_batch_sizes(batch):
    x = np.random.random((batch, 3, 16, 16))

    assert xa.atest(x, post=f"n{batch}") is True
    read_written_image(f"n{batch}")


def test_atest_no_post():
    x = np.random.random((8, 8, 3))

    assert xa.atest(x) is True
    assert os.path.exists("test.png")


def test_atest_raises_when_channels_dim_is_ambiguous():
    x = np.random.random((256, 256, 5))

    with pytest.raises(ValueError):
        xa.atest(x)


@pytest.mark.parametrize("shape", [(256,), (2, 3, 256, 256, 3)])
def test_atest_rejects_wrong_number_of_dims(shape):
    x = np.random.random(shape)

    with pytest.raises(AssertionError):
        xa.atest(x)


def test_tensor():
    class FakeTensor:
        def __init__(self, arr):
            self._arr = arr

        def detach(self):
            return self

        def cpu(self):
            return self

        def __array__(self, dtype=None, copy=None):
            return self._arr

    x = FakeTensor(np.random.random((4, 3, 8, 8)))

    assert xa.atest(x, post="fake_tensor") is True
    read_written_image("fake_tensor")
