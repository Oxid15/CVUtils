import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageChops


def _is_multiple(img):
    shape = img.shape
    if(len(shape) == 1): # (n,) case, array with images of different shapes
        if(len(img[0].shape) > 1): # if contains images and not scalars
            return True
        else:
            raise ValueError(f'Array of scalars was passed with shape={shape} instead of array of images')
    else:
        if(len(shape) == 2): # it is single-channel image (h, w)
            return False
        elif(len(shape) == 3): # it is 3-channel image (h, w, 3) or it is array of single-channel images (n, h, w)
            if(shape[-1] == 3):
                return False
            else:
                return True
        elif(len(shape) == 4): # it is an array of 3-channel images (n, h, w, 3)
            return True


def _show_one(img, figsize, cmap_name, title, suptitle):
    if(figsize == 'auto'):
        figsize = None
    _, ax = plt.subplots(figsize=figsize)
    if(title is not None):
        ax.set_title(title)
    plt.suptitle(suptitle)
    ax.set_title(title)
    ax.imshow(img, cmap=plt.cm.get_cmap(cmap_name))
    plt.show()


def _show_consequent(imgs, shape, figsize, max_img, cmap_name, titles, figsize_factor, suptitle):
    k = 0
    if(shape is None):
        for img in imgs:
            if(k < max_img):
                _, ax = plt.subplots(figsize=figsize)
                plt.suptitle(suptitle)
                ax.imshow(img, cmap=plt.cm.get_cmap(cmap_name))
                if(titles is not None):
                    ax.set_title(titles[k])
                plt.show()
                k += 1


def _show_on_one_plot(imgs, shape, figsize, max_img, cmap_name, titles, figsize_factor, suptitle):
    k = 0
    _, ax = plt.subplots(shape[0], shape[1], figsize=figsize)
    plt.suptitle(suptitle)
    if(shape[0] != 1 and shape[1] != 1):
        for i in range(shape[0]):
            for j in range(shape[1]):
                if(k < max_img):
                    ax[i][j].imshow(imgs[k], cmap=plt.cm.get_cmap(cmap_name))
                    if(titles is not None):
                        ax[i][j].set_title(titles[k])
                    k += 1
    else:
        if(shape[0] == 1):
            opposite_shape = 1
        elif(shape[1] == 1):
            opposite_shape = 0
        for i in range(shape[opposite_shape]):
            if(k < max_img):
                ax[i].imshow(imgs[k], cmap=plt.cm.get_cmap(cmap_name))
                if(titles is not None):
                    ax[i].set_title(titles[k])
                k += 1


def _show_multiple(imgs, shape, figsize, max_img, cmap_name, titles, figsize_factor, suptitle):
    if(max_img is None):
        max_img = len(imgs)
    if(shape == 'auto'):
        shape = _calculate_shape(imgs)
    if(figsize == 'auto'):
        figsize = _calculate_figsize(imgs, shape, figsize_factor)
    if(shape is None):
        _show_consequent(imgs, shape, figsize, max_img, cmap_name, titles, figsize_factor, suptitle)
    else:
        _show_on_one_plot(imgs, shape, figsize, max_img, cmap_name, titles, figsize_factor, suptitle)
    plt.show()


def _show_from(path, shape, figsize, max_img, cmap_name, titles, figsize_factor, suptitle):
    if(os.path.isdir(path)):
        imgs = imread(path, max_img=max_img)
        _show_multiple(imgs, shape, figsize, max_img, cmap_name, titles, figsize_factor, suptitle)
    else:
        img = np.array(Image.open(path))
        imshow(img, figsize=figsize, max_img=max_img, cmap_name=cmap_name, titles=titles, figsize_factor=figsize_factor, suptitle=suptitle)   


def _factorize(num):
    factors = []
    for i in range(1, num + 1):
        if(num % i == 0):
            factors.append(i)
    return np.array(factors)


def _calculate_shape(imgs):
    size = imgs.shape[0]
    height = imgs.shape[1]
    width = imgs.shape[2]

    # euristic for three images to not show them as column
    # when it isn't necessary
    if(size == 3):
        if(height > width):
            return(1, 3)
        else:
            return(3, 1)

    factors = _factorize(size)
    if(len(factors) == 2 and size > 3):
        factors = _factorize(size + 1)
    factors_r = factors[::-1]
    ratios = factors / factors_r
    ratios = ratios - 1.618033988
    arg = np.argmin(np.abs(ratios))
    if(height > width):
        return (factors[arg], factors_r[arg])
    else:
        return (factors_r[arg], factors[arg])


def _calculate_figsize(imgs, shape, factor):
    sum_height = shape[0] * imgs.shape[1]
    sum_width = shape[1] * imgs.shape[2]
    if(factor is None):
        factor = 20
    ratio = sum_height / sum_width
    return (factor, factor * ratio)


def imshow(arg, shape='auto', figsize='auto', figsize_factor=None, max_img=None, cmap_name=None,
    titles=None, suptitle=None, norm=True):
    '''
    Displays an image or a sequence of images.
      
    Parameters:  
    arg: (string, np.ndarray, list, tuple) can be a path to a folder or a file or np.array containing image or sequence of images; if it is a folder path, reads 
    all images from folder, or a certain amount that is set by max_img argument    
      
    shape: default is 'auto', (tuple) if multiple images are passed, determines a shape with which they will be shown in form (rows, columns);
    if set to 'auto' calculates the optimal shape in which images should be shown considering their quantity and aspect ratio; pass 
    tuple to set custom shape; if shape is set for the more images that were passed i.e. (3, 2) for 5 images, it would left the one subplot blank;
    would let one to be blank automatically in 'auto' setting if the number of images passed is prime  
       
    figsize: default is 'auto'; (tuple) figsize argument from matplotlib to set the size of the plot; if set to 'auto' calculates the figsize that would 
    be optimal considering shape and image aspect ratios  
      
    figsize_factor: (int) set this if figsize is set to 'auto'; represents the width of the figure that is passsed to matplotlib; the height is computed by multiplying this value by aspect ratio calculated from shape  
      
    max_img: (int) sets the maximum number of images to read if arg is the path to folder  
      
    cmap_name: (str) the name of colormap from matplotlib  
      
    titles: (list of str) a list of titles for the subplots to the set_title  
      
    suptitle: (str) the title for the whole plot  
      
    norm: (bool) whether to normalize the images by dividing by their np.max  
      
    Returns: None
    '''
    if arg is None:
        raise ValueError("The argument was None!")
    if(type(arg) == str):
        _show_from(arg, shape, figsize, max_img, cmap_name, titles, figsize_factor, suptitle)
    else:
        if(type(arg) == list):
            arg = np.array(arg)
        if _is_multiple(arg):
            if norm:
                for i in range(len(arg)):
                    arg[i] = to_255(arg[i])
            _show_multiple(arg, shape, figsize, max_img, cmap_name, titles, figsize_factor, suptitle)
        else:
            arg = to_255(arg)
            _show_one(arg, figsize, cmap_name, titles, suptitle)


def add_noise_gaussian(img, grayscale, mean=0, std=1):
    '''
    Adds gaussian noise with mean and standard deviation then clips to [0, 255] interval.  
      
    Parameters:  
      
    img: (np.ndarray) input image  
      
    grayscale: (bool) whether image is grayscale  
      
    mean: (float) mean for the gaussian  
      
    std: (float) standard deviation for the gaussian  
      
    Returns: (np.ndarray) new image with noise
    '''
    if(grayscale):
        noise = np.random.normal(mean, std, (img.shape[0], img.shape[1]))
        noisy_img = img + np.dstack((noise, noise, noise))
    else:
        noisy_img = img + np.random.normal(mean, std, img.shape)
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def _read_batch(path, max_img=None, shape=None, sort=False):
        images = []
        filenames = os.listdir(path)
        if sort:
            filenames.sort()
        if(max_img is None):
            max_img = len(filenames)
        for i in range(max_img):
            try:
                image = Image.open(os.path.join(path, filenames[i]))
            except:
                pass
            else:
                images.append(np.array(image))
        
        if(shape is None):
            return np.array(images)
        else:
            return _resize_batch(np.array(images), shape)


def _read_one(path, shape):
    if(shape is None):
        return np.array(Image.open(path))
    else:
        return _resize_one(np.array(Image.open(path)), shape)


def imread(path, max_img=None, shape=None, sort=False):
    '''
    Reads an image or a list of images. Uses PIL: Image.open()  
      
    Parameters:  
      
    path: (str) path to the file or folder  
      
    max_img: (int) maximum number of images  
      
    shape: (tuple) the desired shape for images to be resized; does not resize anything by default, when is set calls PIL.Image.resize() with shape
      
    Returns: (np.ndarray) image or list of images
    '''
    if(os.path.isdir(path)):
        return _read_batch(path, max_img, shape, sort)
    else:
        return _read_one(path, shape)


def open_and_show(path):
    '''
    Reads singular image and shows it by matplotlib.  
      
    Parameters:  
      
    path (str): path to the image file  
      
    Returns: (np.ndarray) image
    '''
    img = imread(path)
    _show_one(img, (10,10), 'viridis', None, 'input')
    return img


def normalize(img):
    '''
    Normalizes the image by subtracting the mean and dividing by the standard deviation.  
      
    Parameters:  
      
    img - image to be normalized  
      
    Returns:  (np.ndarray) normalized image
    '''
    m = np.mean(img)
    std = np.std(img)
    if std != 0:
        img = (img - m) / std
    else:
        img = img - m
    return img


def to_255(img):
    '''
    Normalizes image to [0, 255] by adding minimum and dividing by the maximum value and multiplying by 255. Ignores nans by np.nanmax(). Casts the dtype to uint8.
      
    Parameters:  
      
    img - image to be normalized  
      
    Returns:  (np.ndarray) normalized image of dtype uint8
    '''
    img = img + np.abs(np.min(img))
    _max = np.nanmax(img)
    return ((img / _max) * 255).astype(np.uint8)


def to_1(img):
    '''
    Normalizes image to [0, 1] by dividing by the maximum value. Ignores nans by np.nanmax().
      
    Parameters:  
      
    img - image to be normalized  
      
    Returns:  (np.ndarray) normalized image
    '''
    img = img + np.abs(np.min(img))
    return img / np.nanmax(img)


def _save_one(img, path, name):
    img = Image.fromarray(img)
    try:
        img.save(os.path.join(path, name))
    except Exception:
        raise FileNotFoundError('file {} is not found'.format(os.path.join(path, name))) from BaseException


def save(imgs, path, names):
    '''
    Saves images to the disk.  
      
    Parameters:  
      
    imgs: (np.ndarray, list) singular image or a list of images  
      
    path: (str) path for saving  
      
    names: (list) names of the files; use generator expressions if you have no list of names  
      
    Returns: None
    '''
    if(_is_multiple(imgs)):
        for i in range(len(imgs)):
            _save_one(imgs[i], path, names[i])
    else:
        _save_one(imgs, path, names) 


def _add_suffixes(names, suffixes):
    for i in range(len(names)):
        name, ext = os.path.splitext(names[i])
        names[i] = name + str(suffixes[i]) + ext


def apply(imgs, func, path=None, names=None, verbose=False): #TODO: add support of arg being path to folder
    '''
    Applies function to the images. Can automatically save images when done.   
      
    Parameters:  
      
    imgs: (np.ndarray, list) list of images  
      
    func: (function) function to be applied to images `result = func(img)`  
      
    path: (str) path to folder to save the results
      
    names: (list of str) names of the files; use generator expressions if you have no list of names  
      
    verbose: (bool) False by default: enables indication of progress with time measurement  
      
    Returns: (np.ndarray) results even if saves to folder
    '''
    results = []
    num = 1
    size = len(imgs)
    for img in imgs:
        start = time.time()
        result = func(img)
        results.append(result)
        end = time.time()
        if(verbose):
            print('{}/{} processed in {:.2f}s '.format(num, size, end-start), end= '')
            if(path is not None):
                print('')
        
        if(path is not None):
            start = time.time()
            save(result, path, names[num-1])
            end = time.time()
            if(verbose):
                print('saved in {:.2f}s'.format(end-start))
        num += 1
    return results


def percentile(array):
    '''
    Gives a list of percentiles of array [0, 25 , 50, 75, 100].  
      
    Parameters:  
      
    array: (np.ndarray) array to get percentiles from  
      
    Returns: (pandas.DataFrame) DataFrame with the following percentiles of input array: [0, 25 , 50, 75, 100]. 
    Useful function to call in Jupyter notebooks to fast check the distribution of the particular array. 
    Pandas DataFrame isn't necessary, but it gives nice output in the form of table.  
    '''
    return pd.DataFrame(np.percentile(array, [0, 25, 50, 75, 100]), index=['0', '25', '50', '75', '100']).T


def _resize_batch(imgs, shape, save_aspect, pad):
    res_imgs = np.zeros((len(imgs), shape[0], shape[1], 3), imgs[0].dtype)
    for i in range(len(imgs)):
        res_imgs[i] = _resize_one(imgs[i], shape, save_aspect, pad)
    return res_imgs


def _resize_one(img, shape, save_aspect, pad):
    if(shape == img.shape): #if image already has given shape
        return img

    if save_aspect:
        img = Image.fromarray(img)
        img.thumbnail(shape, Image.ANTIALIAS)
    else:
        img = Image.fromarray(img).resize(shape[::-1])
    
    if pad: 
        crop = img.crop( (0, 0, shape[0], shape[1]) )
        offset_x = max( (shape[0] - img.size[0]) // 2, 0 )
        offset_y = max( (shape[1] - img.size[1]) // 2, 0 )

        img = ImageChops.offset(crop, offset_x, offset_y)
    return np.asarray(img)


def resize(imgs, shape, save_aspect=False, pad=None):
    '''
    Resizes image or a list of images.    
      
    Parameters:  
      
    imgs: (np.ndarray, list) image or a list of images  
      
    shape: desired shape

    save_aspect: if True resizes by the longest side

    pad: if color tuple is given in combination with save_aspect pads the borders 
      
    Returns: (np.ndarray) resized images  
    '''
    if(_is_multiple(imgs)):
        return _resize_batch(imgs, shape, save_aspect, pad)
    else:
        return _resize_one(imgs, shape, save_aspect, pad)
