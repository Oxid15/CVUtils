import os

import pytest

cv2 = pytest.importorskip("cv2")


def read_written_image(post=None):
    name = "test.png" if not post else f"test_{post}.png"
    assert os.path.exists(name)
    img = cv2.imread(name)
    assert img is not None
    return img
