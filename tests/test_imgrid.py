import numpy as np
import pytest

import pxalyze as xa


def test_factorize_prime():
    np.testing.assert_array_equal(xa._factorize(7), [1, 7])


def test_factorize_composite():
    np.testing.assert_array_equal(xa._factorize(12), [1, 2, 3, 4, 6, 12])


def test_factorize_one():
    np.testing.assert_array_equal(xa._factorize(1), [1])


CALCULATE_SHAPE_CASES = [
    (1, (1, 1)),
    (2, (1, 2)),
    (3, (1, 3)),
    (4, (2, 2)),
    (5, (2, 3)),
    (6, (2, 3)),
    (7, (2, 4)),
    (8, (2, 4)),
    (9, (3, 3)),
    (10, (2, 5)),
    (11, (3, 4)),
    (13, (2, 7)),
    (16, (4, 4)),
    (20, (4, 5)),
    (43, (4, 11)),
    (97, (7, 14)),
]


@pytest.mark.parametrize("batch, expected", CALCULATE_SHAPE_CASES)
def test_calculate_shape_square_images(batch, expected):
    x = np.zeros((batch, 3, 10, 10))

    assert xa._calculate_shape(x) == expected


@pytest.mark.parametrize("batch, expected", CALCULATE_SHAPE_CASES)
def test_calculate_shape_covers_the_whole_batch(batch, expected):
    h_count, w_count = expected

    assert h_count * w_count >= batch


@pytest.mark.parametrize("batch", [b for b, _ in CALCULATE_SHAPE_CASES])
def test_calculate_shape_square_images_prefer_landscape(batch):
    x = np.zeros((batch, 3, 10, 10))
    h_count, w_count = xa._calculate_shape(x)

    assert w_count >= h_count


@pytest.mark.parametrize("batch", [1, 2, 3, 4, 5, 6, 7, 10, 11, 16, 20, 43])
def test_imgrid_places_each_image_in_its_own_tile(batch):
    h, w, c = 4, 4, 3
    x = np.zeros((batch, c, h, w))
    for i in range(batch):
        x[i] = i + 1  # unique, distinguishable value per image

    h_count, w_count = xa._calculate_shape(x)
    grid = xa.imgrid(x, head=False)

    assert grid.shape == (c, h * h_count, w * w_count)

    k = 0
    for row in range(h_count):
        for col in range(w_count):
            tile = grid[:, row * h : (row + 1) * h, col * w : (col + 1) * w]
            if k < batch:
                np.testing.assert_array_equal(tile, k + 1)
            else:
                np.testing.assert_array_equal(tile, 0)
            k += 1


def test_imgrid_without_head_has_no_extra_rows():
    x = np.random.random((4, 3, 8, 8))

    grid = xa.imgrid(x, head=False)

    h_count, w_count = xa._calculate_shape(x)
    assert grid.shape == (3, 8 * h_count, 8 * w_count)


def test_imgrid_with_head_reserves_space_for_labels():
    pytest.importorskip("cv2")
    x = np.random.random((4, 3, 8, 8))

    grid = xa.imgrid(x, head=True)

    h_count, w_count = xa._calculate_shape(x)
    assert grid.shape == (3, 8 * h_count + h_count * 16, 8 * w_count)


def test_imgrid_single_image_batch():
    x = np.random.random((1, 3, 5, 5))

    grid = xa.imgrid(x, head=False)

    assert grid.shape == (3, 5, 5)
    np.testing.assert_array_equal(grid, x[0])
