import os

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

def show_one(img, figsize):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()

def show_multiple(imgs, shape, figsize, max_img):
    k = 0

    if(shape is None):
        plt.figure(figsize=figsize)
        for img in imgs:
            if(k < max_img):
                plt.imshow(img)
                plt.show()
                k += 1
    else:
        _, ax = plt.subplots(shape[0], shape[1], figsize=figsize)
        if(shape[0] != 1 and shape[1] != 1):
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if(k < imgs.shape[0] or k < max_img):
                        ax[i][j].imshow(imgs[k])
                        k += 1
        else:
            if(shape[0] == 1):
                opposite_shape = 1
            elif(shape[1] == 1):
                opposite_shape = 0
            for i in range(shape[opposite_shape]):
                if(k < imgs.shape[0] or k < max_img):
                    ax[i].imshow(imgs[k])
                    k += 1
        plt.show()

def show_from(path, shape, figsize, max_img):
    if(os.path.isdir(path)):
        imgs = read_images(path, max_img=max_img)
        show_multiple(imgs, shape, figsize, max_img)
    else:
        img = np.array(Image.open(path))
        imshow(img, figsize=figsize)       

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
    k = 0
    images = []
    filenames = os.listdir(path)
    for name in filenames:
        if(k < max_img):
            try:
                image = Image.open(os.path.join(path, name))
            except Exception:
                pass
            else:
                images.append(np.array(image))
                k += 1
    return np.array(images)

def imshow(arg, shape=None, figsize=None, max_img=None):
    if(type(arg) == str):
        show_from(arg, shape, figsize, max_img)
    else:
        if(len(arg.shape) <= 3):
            show_one(arg, figsize)
        elif(len(arg.shape) == 4):
            show_multiple(arg, shape, figsize, max_img)

def add_noise_gaussian(img, mean, std, grayscale=False):
    if(grayscale):
        noise = np.random.normal(mean, std, (img.shape[0], img.shape[1]))
        noisy_img = img + np.dstack((noise, noise, noise))
    else:
        noisy_img = img + np.random.normal(mean, std, img.shape)
    return np.clip(noisy_img, 0, 255).astype(np.uint8)