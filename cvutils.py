from cvutils_import import *

def _is_multiple(img):
    shape = img.shape
    if(len(shape) == 1): # (n,) case, array with images of different shapes
        if(len(img[0].shape) > 1): # if contains images and not scalars
            return True
        else:
            raise ValueError('Array of scalars was passed with shape={shape} instead of array of images')
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

def _show_one(img, figsize, cmap, title, suptitle):
    if(figsize == 'auto'):
        figsize = None
    _, ax = plt.subplots(figsize=figsize)
    if(title is not None):
        ax.set_title(title)
    plt.suptitle(suptitle)
    ax.set_title(title)
    ax.imshow(img, cmap=cmap)
    plt.show()

def _show_consequent(imgs, shape, figsize, max_img, cmap, titles, figsize_factor, suptitle):
    k = 0
    if(shape is None):
        for img in imgs:
            if(k < max_img):
                _, ax = plt.subplots(figsize=figsize)
                ax.imshow(img)
                if(titles is not None):
                    ax.set_title(titles[k])
                plt.show()
                k += 1

def _show_on_one_plot(imgs, shape, figsize, max_img, cmap, titles, figsize_factor, suptitle):
    k = 0
    _, ax = plt.subplots(shape[0], shape[1], figsize=figsize)
    if(shape[0] != 1 and shape[1] != 1):
        for i in range(shape[0]):
            for j in range(shape[1]):
                if(k < max_img):
                    ax[i][j].imshow(imgs[k], cmap=cmap)
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
                ax[i].imshow(imgs[k], cmap=cmap)
                if(titles is not None):
                        ax[i].set_title(titles[k])
                k += 1

def _show_multiple(imgs, shape, figsize, max_img, cmap, titles, figsize_factor, suptitle):
    plt.suptitle(suptitle)
    
    if(max_img is None):
        max_img = len(imgs)
    if(shape == 'auto'):
        shape = _calculate_shape(imgs)
    if(figsize == 'auto'):
        figsize = _calculate_figsize(imgs, shape, figsize_factor)
        if(shape is None):
            _show_consequent(imgs, shape, figsize, max_img, cmap, titles, figsize_factor, suptitle)
        else:
            _show_on_one_plot(imgs, shape, figsize, max_img, cmap, titles, figsize_factor, suptitle)
        plt.show()

def show_from(path, shape, figsize, max_img, cmap, titles, figsize_factor, suptitle):
    if(os.path.isdir(path)):
        imgs = imread(path, max_img=max_img)
        _show_multiple(imgs, shape, figsize, max_img, cmap, titles, figsize_factor, suptitle)
    else:
        img = np.array(Image.open(path))
        imshow(img, figsize=figsize, max_img=max_img, cmap=cmap, titles=titles, figsize_factor=figsize_factor, suptitle=suptitle)   

# def read_images_sort(path):
#     images = []
#     filenames = os.listdir(path)
#     #sorting as integers
#     filenames = [int(x[:-4]) for x in filenames]
#     filenames.sort()
#     #TODO: expand to any format
#     filenames = [str(x)+'.bmp' for x in filenames]
#     for name in filenames:
#         image = Image.open(path + name)
#         images.append(np.array(image))
#     return np.array(images)

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

    #euristic
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

def imshow(arg, shape='auto', figsize='auto', figsize_factor=None, max_img=None, cmap=None, titles=None, suptitle=None):
    if arg is None:
        raise ValueError("The argument was None!")
    if(type(arg) == str):
        show_from(arg, shape, figsize, max_img, cmap, titles, figsize_factor, suptitle)
    else:
        if(type(arg) == list):
            arg = np.array(arg)
        if(_is_multiple(arg)):
           _show_multiple(arg, shape, figsize, max_img, cmap, titles, figsize_factor, suptitle)
        else:
           _show_one(arg, figsize, cmap, titles, suptitle)

def add_noise_gaussian(img, mean=0, std=1, grayscale=False):
    if(grayscale):
        noise = np.random.normal(mean, std, (img.shape[0], img.shape[1]))
        noisy_img = img + np.dstack((noise, noise, noise))
    else:
        noisy_img = img + np.random.normal(mean, std, img.shape)
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def imread(path, max_img=None):
    if(os.path.isdir(path)):
        images = []
        filenames = os.listdir(path)
        if(max_img is None):
            max_img = len(filenames)
        for i in range(max_img):
            try:
                image = Image.open(os.path.join(path, filenames[i]))
            except Exception:
                pass
            else:
                images.append(np.array(image))
        return np.array(images)
    else:
        return np.array(Image.open(path))

def open_and_show(path):
    img = open(path)
    _show_one(img, (10,10), plt.cm.get_cmap('viridis'), 'input', 'input')
    return img

def normalize(img):
    return (img - np.mean(img)) / np.std(img)

def to_255(image):
    _max = np.nanmax(image)
    return ((image / _max) * 255).astype(np.uint8)

def to_1(image):
    return image / np.nanmax(image)

def _save_one(img, path, name):
    img = Image.fromarray(img)
    try:
        img.save(os.path.join(path, name))
    except Exception:
        raise FileNotFoundError('file {} is not found'.format(os.path.join(path, name))) from BaseException

def save(imgs, path, names):
    if(_is_multiple(imgs)):
        for i in range(len(imgs)):
            _save_one(imgs[i], path, names[i])
    else:
        _save_one(imgs, path, names) 

def _add_suffixes(names, suffixes):
    for i in range(len(names)):
        name, ext = os.path.splitext(names[i])
        names[i] = name + str(suffixes[i]) + ext

def apply(imgs, func, save_path=None, names=None, verbose=False): #TODO: add support of arg being path to folder
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
            if(save_path is not None):
                print('')
        
        if(save_path is not None):
            start = time.time()
            save(result, save_path, names[num-1])
            end = time.time()
            if(verbose):
                print('saved in {:.2f}s'.format(end-start))
        num += 1

    return results

def percentile(array):
    return pd.DataFrame(np.percentile(array, [0, 25, 50, 75, 100]), index=['0', '25', '50', '75', '100']).T

def _resize_batch(imgs, shape):
    res_imgs = np.zeros((len(imgs), shape[0], shape[1]), imgs[0].dtype)
    for i in range(len(imgs)):
        res_imgs[i] = _resize_one(imgs[i], shape)
    return res_imgs

def _resize_one(img, shape):
    if(shape == img.shape): #if image already has given shape
        return img
    return cv2.resize(img, shape)

def resize(imgs, shape):
    if(_is_multiple(imgs)):
        return _resize_batch(imgs, shape)
    else:
        return _resize_one(imgs, shape)