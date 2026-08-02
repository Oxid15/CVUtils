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
        (3, 256, 256),  # CHW, single image
        (13, 256, 256),  # CHW, many channels
        (1, 3, 256, 256),  # BCHW, batch of 1
        (10, 3, 256, 256),  # BCHW, batch of 10
        (10, 13, 256, 256),  # BCHW, batch of 10, many channels
    ],
)
def test_featest_handles_a_variety_of_shapes(shape):
    x = np.random.random(shape)

    assert xa.featest(x, post="shape") is True


def test_featest_normalizes_anyway():
    x = np.random.random((3, 256, 256)) * 0.01

    assert xa.featest(x, post="small") is True

    img = read_written_image("small")
    assert img.min() == 0
    assert img.max() == 255


def test_featest_handles_batch_and_channels_the_same():
    x = np.random.random((43, 3, 32, 32))
    y = np.random.random((3, 43, 32, 32))

    assert xa.featest(x, post="x") is True
    assert xa.featest(y, post="y") is True

    x = read_written_image("x")
    y = read_written_image("y")

    assert x.shape == y.shape


@pytest.mark.parametrize("batch", [1, 2, 5, 7, 11, 20])
def test_featest_arbitrary_batch_sizes(batch):
    x = np.random.random((batch, 3, 16, 16))

    assert xa.atest(x, post=f"n{batch}") is True
    read_written_image(f"n{batch}")


@pytest.mark.parametrize("shape", [(256,), (2, 3, 256, 256, 3)])
def test_featest_rejects_wrong_number_of_dims(shape):
    x = np.random.random(shape)

    with pytest.raises(AssertionError):
        xa.atest(x)
