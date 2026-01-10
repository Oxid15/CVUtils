"""
Copyright 2025-2026 Ilia Moiseev

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__version__ = "0.1.0"
__author__ = "Ilia Moiseev"
__author_email__ = "ilia.moiseev.5@yandex.ru"

from pprint import pprint as pp

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

__version__ = "0.1.0"

PHI = 1.618033988
CHANNEL_OPTIONS = (1, 3)


def what(x):
    """
    Introspection tool designed for debugging in REPL mode or with prints

    Parameters
    ----------
    x : Any
        Best options are numpy.ndarray, torch.tensor, list, dict

    Returns
    -------
    Dict[str, Any]
        Dictionary with useful fields for debugging arrays
    """
    d = {"type": type(x)}

    for method in ["min", "mean", "max"]:
        if hasattr(x, method):
            d[method] = getattr(x, method)()

    if hasattr(x, "shape"):
        d["shape"] = x.shape

    if hasattr(x, "dtype"):
        d["dtype"] = x.dtype

    if isinstance(x, dict):
        d["keys"] = list(x.keys())

    if isinstance(x, list):
        d["len"] = len(x)

    return d


def rwhat(x):
    if isinstance(x, (list, tuple)):
        return [rwhat(item) for item in x]
    elif isinstance(x, dict):
        return {k: rwhat(x[k]) for k in x.keys()}
    else:
        return what(x)


def tonp(x):
    """
    Converts input to numpy array, works best with torch.Tensor
    """
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu()
    return np.array(x)


def to1(x):
    """
    Min max normalization for arrays (should work for everything with .min() and .max())

    Parameters
    ----------
    x : Any
        Array or tensor

    Returns
    -------
    Any
        Normalized array
    """
    min_val = x.min()
    max_val = x.max()

    if min_val == max_val:
        print(
            f"Failed to normalize: min_val == max_val == {min_val} falling back to zeros"
        )
        return np.zeros_like(x)

    return (x - min_val) / (max_val - min_val)


def to255(x):
    """
    Does the same thing as to1() but also multiplies by 255
    handy for standard cv2.imwrite

    Parameters
    ----------
    x : Any
        Array with any range

    Returns
    -------
    Any
        Array with range [0, 255]
    """
    return to1(x) * 255


# def mplot(x): ...


# def mhist(x): ...


def _factorize(num):
    factors = []
    for i in range(1, num + 1):
        if num % i == 0:
            factors.append(i)
    return np.array(factors)


def _calculate_shape(imgs):
    size = imgs.shape[0]
    height = imgs.shape[2]
    width = imgs.shape[3]

    factors = _factorize(size)
    if len(factors) == 2 and size > 3:
        factors = _factorize(size + 1)
    factors_r = factors[::-1]
    ratios = factors / factors_r
    ratios = ratios - PHI
    arg = np.argmin(np.abs(ratios))
    
    y = factors[arg]
    x = factors_r[arg]
    
    # swap sides if the image is vertical
    if height * y > width * x:
        x, y = y, x
    
    return y, x


def imgrid(x, head=False):
    """
    The utility for tiling the batch of images in the optimal shape
    will create the tiling with the shape
    closest to the golden ratio (horizontal image)

    Parameters
    ----------
    x : Any
        Images of shape (B, C, H, W)

    Returns
    -------
    Any
        Tiled images of shape (C, nH, nW)
    """
    _, c, h, w = x.shape
    h_count, w_count = _calculate_shape(x)

    k = 0
    text_h_px = 16 if cv2 is not None and head else 0
    grid = np.zeros(
        (h * h_count + h_count * text_h_px, w * w_count, c)
    )  # different dim order - (h, w, c) for opencv
    for i in range(h_count):
        for j in range(w_count):
            if k >= len(x):
                break

            grid[
                i * h + (i + 1) * text_h_px : (i + 1) * h + (i + 1) * text_h_px,
                j * w : (j + 1) * w,
                :,
            ] = x[k].transpose(1, 2, 0)
            k += 1

    if cv2 is not None and head:
        k = 0
        color = (1, 1, 1)
        for i in range(h_count):
            for j in range(w_count):
                if k >= len(x):
                    break

                cv2.putText(
                    grid,
                    f"{k:0>4d}",
                    (j * w, i * (h + text_h_px) + text_h_px - 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,
                    color,
                    1,
                )
                k += 1

    return grid.transpose(2, 0, 1)


def _find_channels(x):
    for i, dim in enumerate(x.shape[::-1]):
        if dim in CHANNEL_OPTIONS:
            return len(x.shape) - 1 - i
    return None


def test(x, post=None):
    """
    Just a wrapper to replace cv2.imwrite("test.png", x)
    with test(x)

    Parameters
    ----------
    x : Any
        Input image
    post : str, optional
        string to add to filename, by default None

    Returns
    -------
    bool
        The output of cv2.imwrite
    """
    name = "test"
    if post:
        name = f"{name}_{post}"

    return cv2.imwrite(f"{name}.png", x)


def atest(x, post=None):
    """
    Automatically visualize images by dumping them on disk
    will save the file test.png in the current directory

    Input can be a single image in the format (C, H, W)
    or a batch of images (B, C, H, W) in this case it will tile
    each image in the batch to make it (C, H, W) for output

    will minmax normalize images for visualization, see to1

    Parameters
    ----------
    x : Any
        numpy.array or torch.Tensor of images

    post : str, optional
        The string to append after the filename, by default None

    Returns
    -------
    bool
        The output of cv2.imwrite

    Raises
    ------
    ValueError
        If cannot find which dimension is channels, see CHANNEL_OPTIONS
    """

    x = tonp(x)

    shape = x.shape
    assert len(shape) in (3, 4)
    channels_idx = _find_channels(x)

    if channels_idx is None:
        raise ValueError(
            f"Cannot find channels dim in {shape}, should be in {CHANNEL_OPTIONS}"
        )

    dim_order = [i for i in range(len(shape))]
    if len(shape) == 4:
        dim_order = [
            dim_order[0],
            channels_idx,
            *[i for i in dim_order[1:] if i != channels_idx],
        ]
    else:
        dim_order = [channels_idx, *[i for i in dim_order if i != channels_idx]]
    x = x.transpose(*dim_order)

    if len(shape) == 4:
        x = imgrid(x, head=True)

    # TODO: imgrid result can be huge - need to add max_size

    assert shape[channels_idx] in CHANNEL_OPTIONS

    x = x.transpose(1, 2, 0)

    if (x > 1).any():
        x = to1(x)

    name = "test"
    if post:
        name = f"{name}_{post}"

    return cv2.imwrite(f"{name}.png", x * 255)


def lm(x, *f):
    """
    lm - short for list map

    Sometimes it is an easier way to
    chain functions in debug console like this:
    lm(x, tonp, to1, lambda x: cv2.blur(x, (3, 3)))
    the same as:
    cv2.blur(to1(tonp(x)), (3, 3))

    x can be a single item or a list (or tuple)
    >>> lm([1, 2], lambda x: x + 1)
    [2, 3]

    >>> lm([1, 2], lambda x: x + 1, lambda x: x + 1)
    [3, 4]
    """
    if isinstance(x, (list, tuple)):
        for func in f:
            x = list(map(func, x))
    else:
        for func in f:
            x = func(x)
    return x
