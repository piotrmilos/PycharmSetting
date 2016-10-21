def set_matplotlib_backend():
    import matplotlib
    import os
    v = os.environ.get('MATPLOTLIB_BACKEND')
    print v
    if v:
        matplotlib.use(v)

set_matplotlib_backend()

import os
import scipy
from scipy.misc import imresize
import numpy as np


import matplotlib.pyplot as plt

__author__ = 'maciek'

def plot_image_to_file(img, filepath, interpolation='none'):
    plt.imshow(img, cmap='gray', interpolation=interpolation)
    plt.savefig(filepath)
    plt.clf()

def plot_image_to_file_gray(img, filepath, interpolation='none'):
    plt.imshow(img, cmap='gray', interpolation=interpolation)
    plt.savefig(filepath)
    plt.clf()

def plot_image_to_file_rgb(img, filepath, interpolation='none'):
    plt.imshow(img, interpolation=interpolation)
    plt.savefig(filepath)
    plt.clf()

def numpy_img_to_PIL(img):
    from PIL import Image
    if img.dtype in ['float32', 'float64']:
        data = np.asarray(img * 255, dtype=np.uint8)
    elif img.dtype in ['int32', 'uint8', 'uint32']:
        data = np.asarray(img, dtype=np.uint8)
    else:
        raise RuntimeError('Unknown dtype ' + str(img.dtype))

    if len(img.shape) == 3 and img.shape[2] == 3:
        im = Image.fromarray(data, 'RGB')
    elif len(img.shape) == 2:
        im = Image.fromarray(data, 'L')
    else:
        raise RuntimeError()
    return im



def plot_image_to_file2(img, filepath, interpolation='none', only_image=True, driver='JPEG'):
    if only_image:
        im = numpy_img_to_PIL(img)
        im.save(filepath)
        return filepath
    else:
        plt.imshow(img, interpolation=interpolation)
        plt.savefig(filepath)
        return filepath



def plot_image(img, cmap='gray', interpolation='none'):
    plt.imshow(img, cmap=cmap, interpolation=interpolation)
    plt.show()
    plt.clf()


def resize_crop(img, inter_size):

    # img should be in HxWxC
    assert (inter_size % 2 == 0)


    h = img.shape[0]
    w = img.shape[1]

    ratio = h / float(w)

    # We resize so that the smaller side has length equal to inter, and crop center square inter x inter.
    if h < w:
        newshape = (inter_size, int(inter_size / ratio), img.shape[2])
        img = my_imresize(img, newshape)[:, (newshape[1] / 2) - (inter_size / 2): (newshape[1] / 2) + (inter_size / 2), :]
    else:
        newshape = (int(inter_size * ratio), inter_size, img.shape[2])
        img = my_imresize(img, newshape)[(newshape[0] / 2) - (inter_size / 2): (newshape[0] / 2) + (inter_size / 2), :, :]
    return img

def resize_simple(img, inter_size):

    # img should be in HxWxC
    assert (inter_size % 2 == 0)
    return my_imresize(img, (inter_size, inter_size))


def fetch_path_local(path):
    # This returns images in HxWxC format, dtyep = uint8 probably
    img = scipy.misc.imread(path)

    if len(img.shape) == 2:
        # Some images are in grayscale
        img = img.reshape(img.shape[0], img.shape[1], 1)
        img = img.repeat(3, axis=2)

    if img.shape[2] > 3:
        # Some images have more than 3-channels. Doing same thing as fbcunn
        img = img[:, :, :3]
    return img


def my_imresize(img, newshape):
    if newshape == img.shape:
        return img
    else:
        return imresize(img, newshape)