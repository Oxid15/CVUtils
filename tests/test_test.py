import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import pxalyze as xa


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_test_writes_default_filename():
    x = np.random.randint(0, 255, (16, 16, 3)).astype(np.uint8)

    assert xa.test(x) is True
    assert os.path.exists("test.png")


def test_test_writes_filename_with_post_suffix():
    x = np.random.randint(0, 255, (16, 16, 3)).astype(np.uint8)

    assert xa.test(x, "a") is True
    assert os.path.exists("test_a.png")
    assert not os.path.exists("test.png")


def test_test_does_not_normalize():
    x = np.zeros((4, 4, 3), dtype=np.uint8)
    x[0, 0] = 255

    xa.test(x)

    img = cv2.imread("test.png")
    assert img[0, 0].tolist() == [255, 255, 255]
    assert img[1, 1].tolist() == [0, 0, 0]
