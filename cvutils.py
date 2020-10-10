from cvutils_import import *

def show_one(img, figsize, cmap):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()

def show_multiple(imgs, shape, figsize, max_img, cmap):
    k = 0
    if(max_img is None):
        max_img = len(imgs)

    if(shape is None):
        for img in imgs:
            if(k < max_img):
                plt.figure(figsize=figsize)
                plt.imshow(img)
                plt.show()
                k += 1
    else:
        _, ax = plt.subplots(shape[0], shape[1], figsize=figsize)
        if(shape[0] != 1 and shape[1] != 1):
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if(k < max_img):
                        ax[i][j].imshow(imgs[k], cmap=cmap)
                        k += 1
        else:
            if(shape[0] == 1):
                opposite_shape = 1
            elif(shape[1] == 1):
                opposite_shape = 0
            for i in range(shape[opposite_shape]):
                if(k < max_img):
                    ax[i].imshow(imgs[k], cmap=cmap)
                    k += 1
        plt.show()

def show_from(path, shape, figsize, max_img, cmap):
    if(os.path.isdir(path)):
        imgs = read_images(path, max_img=max_img)
        show_multiple(imgs, shape, figsize, max_img, cmap)
    else:
        img = np.array(Image.open(path))
        imshow(img, figsize=figsize, cmap=cmap)       

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

def read_images(path, max_img):
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

def imshow(arg, grayscale=False, shape=None, figsize=None, max_img=None, cmap=None):
    if arg is None:
        raise ValueError("The argument was None!")
    if(type(arg) == str):
        show_from(arg, shape, figsize, max_img, cmap)
    else:
        if(type(arg) == list):
            arg = np.array(arg)
        if(grayscale and len(arg.shape) == 3 or len(arg.shape) == 4):
            show_multiple(arg, shape, figsize, max_img, cmap)
        elif(len(arg.shape) <= 3):
            show_one(arg, figsize, cmap)

def add_noise_gaussian(img, mean, std, grayscale=False):
    if(grayscale):
        noise = np.random.normal(mean, std, (img.shape[0], img.shape[1]))
        noisy_img = img + np.dstack((noise, noise, noise))
    else:
        noisy_img = img + np.random.normal(mean, std, img.shape)
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def open(path):
    return np.array(Image.open(path))

def normalize_to_255(image):
    _max = np.nanmax(image)
    return ((image / _max) * 255).astype(np.uint8)

def normalize_to_1(image):
    return image / np.nanmax(image)