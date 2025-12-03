from typing import Any, Dict

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


def what(x: Any) -> Dict[str, Any]:
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


def tonp(x):
    """
    Converts input to numpy array, works best with torch.Tensor
    """
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu()
    return np.array(x)


def to1(x: Any) -> Any:
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

    return (x - min_val) / (max_val - min_val)


def to255(x: Any) -> Any:
    """
    Does the same thing as to1() but also multiplies by 255

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


def mplot(x): ...


def mhist(x): ...


def _factorize(num):
    factors = []
    for i in range(1, num + 1):
        if num % i == 0:
            factors.append(i)
    return np.array(factors)


def _calculate_shape(imgs):
    size = imgs.shape[0]
    height = imgs.shape[1]
    width = imgs.shape[2]

    # heuristic for three images to not show them as column
    # when it isn't necessary
    if size == 3:
        if height > width:
            return (1, 3)
        else:
            return (3, 1)

    factors = _factorize(size)
    if len(factors) == 2 and size > 3:
        factors = _factorize(size + 1)
    factors_r = factors[::-1]
    ratios = factors / factors_r
    ratios = ratios - PHI
    arg = np.argmin(np.abs(ratios))
    if height > width:
        return (factors[arg], factors_r[arg])
    else:
        return (factors_r[arg], factors[arg])


def imgrid(x):
    _, c, h, w = x.shape
    h_count, w_count = _calculate_shape(x)

    k = 0
    text_h_px = 10 if cv2 is not None else 0
    grid = np.zeros((c, h * h_count + h_count * text_h_px, w * w_count))
    for i in range(h_count):
        for j in range(w_count):
            grid[
                :,
                i * h + (i + 1) * text_h_px : (i + 1) * h + (i + 1) * text_h_px,
                j * w : (j + 1) * w,
            ] = x[k]
            k += 1

    # if cv2 is not None:
    #     k = 0
    #     color = (255, 255, 255)
    #     for i in range(h_count):
    #         for j in range(w_count):
    #             cv2.putText(
    #                 grid,
    #                 f"{k:0>5d}",
    #                 (i * h, j * w),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 10,
    #                 color,
    #                 1,
    #             )
    #             k += 1

    return grid


def _find_channels(x):
    for i, dim in enumerate(x.shape[::-1]):
        if dim in CHANNEL_OPTIONS:
            return len(x.shape) - 1 - i
    return None


def atest(x, post=None):
    x = tonp(x)

    shape = x.shape
    assert len(shape) in (3, 4)
    channels_idx = _find_channels(x)

    if channels_idx is None:
        raise ValueError(f"Cannot find channels dim in {shape}, should be in {CHANNEL_OPTIONS}")

    dim_order = [i for i in range(len(shape))]
    if len(shape) == 4:
        dim_order = [dim_order[0], channels_idx, *[i for i in dim_order[1:] if i != channels_idx]]
    else:
        dim_order = [channels_idx, *[i for i in dim_order if i != channels_idx]]
    x = x.transpose(*dim_order)

    if len(shape) == 4:
        x = imgrid(x)

    assert shape[channels_idx] in CHANNEL_OPTIONS

    x = x.transpose(1, 2, 0)

    if (x > 1).any():
        x = to1(x)

    name = "test"
    if post:
        name = f"{name}_{post}"

    return cv2.imwrite(f"{name}.png", x * 255)


# def _resize_batch(imgs, shape, save_aspect, pad):
#     res_imgs = np.zeros((len(imgs), shape[0], shape[1], 3), imgs[0].dtype)
#     for i in range(len(imgs)):
#         res_imgs[i] = _resize_one(imgs[i], shape, save_aspect, pad)
#     return res_imgs


# def _resize_one(img, shape, save_aspect, pad):
#     if shape == img.shape:  # if image already has given shape
#         return img

#     if save_aspect:
#         img = Image.fromarray(img)
#         img.thumbnail(shape, Image.ANTIALIAS)
#     else:
#         img = Image.fromarray(img).resize(shape[::-1])

#     if pad:
#         crop = img.crop((0, 0, shape[0], shape[1]))
#         offset_x = max((shape[0] - img.size[0]) // 2, 0)
#         offset_y = max((shape[1] - img.size[1]) // 2, 0)

#         img = ImageChops.offset(crop, offset_x, offset_y)
#     return np.asarray(img)


# def resize(imgs, shape, save_aspect=False, pad=None):
#     # """
#     # Resizes image or a list of images.

#     # Parameters:

#     # imgs: (np.ndarray, list) image or a list of images

#     # shape: desired shape

#     # save_aspect: if True resizes by the longest side

#     # pad: if color tuple is given in combination with save_aspect pads the borders

#     # Returns: (np.ndarray) resized images
#     # """
#     if _is_multiple(imgs):
#         return _resize_batch(imgs, shape, save_aspect, pad)
#     else:
#         return _resize_one(imgs, shape, save_aspect, pad)


def test(x) -> bool: ...


def lm(x, *f):
    """
    Sometimes it is an easier way to
    chain functions in debug console
    """
    if isinstance(x, list):
        for func in f:
            x = list(map(func, x))
    else:
        for func in f:
            x = func(x)
    return x
