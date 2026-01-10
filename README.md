# Pxalyze

pxalyze - a single file for your computer vision debugging

## Installation

`pxalyze` tries to be very universal so it should work with Python since 3.7 (the lowest tested version)

```bash
pip install pxalyze
```

It requires `numpy` and optionally `opencv` for visualizations
you can install full version with

```bash
pip install pxalyze[opencv]
```

## Usage

`pxalyze` is a single file library for debugging images as arrays or tensors
it is perfect for interactive debugging in REPL environments such as debug consoles
or just interactive python shells

The intended usage is the following, but depending on your setup you may choose to import it differently

```python
from pxalyze import *
```

Alternatively to isolate the namespaces you can also use

```python
import pxalyze as cvu
```

### Introspection

`what` is the method for introspection that helps to quickly analyze arrays or tensors, it returns dict

```python
>>> from pxalyze import *
>>> x = np.random.random((1, 3, 256, 256))
>>> what(x)
{'type': <class 'numpy.ndarray'>, 'min': 1.072805627044815e-05, 'mean': 0.5001992238233383, 'max': 0.9999899398862, 'shape': (1, 3, 256, 256), 'dtype': dtype('float64')}
```

It works with standard types too, displaying basic information

```python
>>> what([1, 2])
{'type': <class 'list'>, 'len': 2}

>>> what({"a": 0})
{'type': <class 'dict'>, 'keys': ['a']}
```

You can also use `rwhat` to recursively introspect lists or dicts of arrays

```python
>>> rwhat([np.array(1), np.array(2)])
[{'type': <class 'numpy.ndarray'>, 'min': 1, 'mean': 1.0, 'max': 1, 'shape': (), 'dtype': dtype('int64')}, {'type': <class 'numpy.ndarray'>, 'min': 2, 'mean': 2.0, 'max': 2, 'shape': (), 'dtype': dtype('int64')}]
```

When you `import *` from `pxalyze` you also get `pprint` imported as `pp` so you can use

```python
>>> from pxalyze import *
>>> pp(rwhat([np.array(1), np.array(2)]))
[{'dtype': dtype('int64'),
  'max': 1,
  'mean': 1.0,
  'min': 1,
  'shape': (),
  'type': <class 'numpy.ndarray'>},
 {'dtype': dtype('int64'),
  'max': 2,
  'mean': 2.0,
  'min': 2,
  'shape': (),
  'type': <class 'numpy.ndarray'>}]
```

### Visualization

`test` and `atest` are the function that take tensors or arrays and write a visualization on disk

`test` is a simple wrapper around `cv2.imwrite`, so it is very limited, but reliable

```python
>>> test(np.random.randint(0, 255, (256, 256, 3)))
True
```

This will save the noisy image "test.png" to the current folder

If you are saving more than one images for comparison just do

```python
a = np.random.randint(0, 255, (256, 256, 3))
b = np.random.randint(0, 255, (256, 256, 3))

test(a, "a")
test(b, "b")
```

This saves two files named "test_a.png" and "test_b.png" which is more convenient
than type `cv2.imwrite("test_a.png", a)` each time

`atest` is more powerful - it can accept wide range of shapes and ranges and visualize

```python
>>> a = np.random.random((256, 256, 3))
>>> b = np.random.random((3, 256, 256))
>>> c = np.random.random((1, 3, 256, 256))
>>> d = np.random.random((10, 3, 256, 256))
>>> e = np.random.random((10, 256, 256, 3)) * 10000

>>> atest(a, "a")
True
>>> atest(b, "b")
True
>>> atest(c, "c")
True
>>> atest(d, "d")
True
>>> atest(e, "e")
True
```

In case when there is a batch dimension in the tensor `atest`
will try to tile the images into a single one and display them
while writing index of each image above it

```python
>>> atest(np.random.random((43, 3, 32, 32)))
True
```

will produce the following image

![atest](images/atest.png)

### Utilities

Use `to1` and `to255` to minmax normalize arrays

```python
>>> a = np.random.random((1, 2)) * 100 + 25
>>> what(to1(a))
{'type': <class 'numpy.ndarray'>, 'min': 0.0, 'mean': 0.5, 'max': 1.0, 'shape': (1, 2), 'dtype': dtype('float64')}

>>> what(to1(np.ones((1, 2))))
Failed to normalize: min_val == max_val == 1.0 falling back to zeros
{'type': <class 'numpy.ndarray'>, 'min': 0.0, 'mean': 0.0, 'max': 0.0, 'shape': (1, 2), 'dtype': dtype('float64')}
```

Use `imgrid` to manually tile the batch of images (this is what `atest` does)

```python
>>> what(imgrid(np.random.random((3, 3, 25, 25))))
{'type': <class 'numpy.ndarray'>, 'min': 0.00037330399391755087, 'mean': 0.4978226227728666, 'max': 0.9999395359165014, 'shape': (3, 25, 75), 'dtype': dtype('float64')}
```

You can use `lm` to conveniently chain function in terminal without the need to match all the brackets

```python
>>> a = np.random.random((10, 3, 16, 16))
>>> what(lm(a, tonp, to1, np.abs, imgrid))
{'type': <class 'numpy.ndarray'>, 'min': 0.0, 'mean': 0.4960013939669902, 'max': 1.0, 'shape': (3, 32, 80), 'dtype': dtype('float64')}
```

is the same as

```python
what(imgrid(np.abs(to1(tonp(a)))))
{'type': <class 'numpy.ndarray'>, 'min': 0.0, 'mean': 0.4960013939669902, 'max': 1.0, 'shape': (3, 32, 80), 'dtype': dtype('float64')}
```

but much less brackets!

> It actually saved me a lot of manual debugging time since I don't need to waste time on trying to find a single unmatched bracket in the REPL

You can use `tonp` to conventiently convert torch.Tensors to numpy and not type `.detach().cpu().numpy()` ever again.

```python
>>> from pxalyze import *
>>> a = torch.randn((4, 3, 16, 16)).to("cuda:0")
>>> what(tonp(a))
{'type': <class 'numpy.ndarray'>, 'min': -3.2309637, 'mean': -0.010580406, 'max': 3.5923967, 'shape': (4, 3, 16, 16), 'dtype': dtype('float32')}

`atest` does is automatically

```python
>>> atest(a)
True
```

## Contributing

Pull requests and issues are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)

## Versions

This project uses Semantic Versioning - <https://semver.org/>
