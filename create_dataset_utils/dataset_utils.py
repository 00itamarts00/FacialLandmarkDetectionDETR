import glob
import os
from typing import Callable, Any

import matplotlib.pyplot as plt
import numpy as np
from imgaug.augmentables import Keypoint, KeypointsOnImage
from numpy import ndarray
from skimage.io import imread, imsave
import menpo.io as mio
from create_dataset_utils.txt_parser import get_pts_from_txt
from utils.file_handler import FileHandler

round2int: Callable[[Any], ndarray] = lambda x: np.round(x).astype(int)
round2natural: Callable[[Any], ndarray] = lambda x: np.clip(np.round(x), a_min=0, a_max=np.inf).astype(int)


def read_pair_img_pts(basename):
    image = read_img(basename)
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    pts_tuple = get_pts_from_txt(f'{basename}.pts')
    class_labels = range(1, len(pts_tuple) + 1)
    kps = [Keypoint(x=pts[0], y=pts[1]) for pts in pts_tuple]
    kps_oi = KeypointsOnImage(kps, shape=image.shape)
    return image, kps_oi


def all_pts_files_in_dataset(dataset_path):
    files = glob.glob(dataset_path + '/**/*.pts', recursive=True)
    relpaths = [os.path.relpath(i, dataset_path).replace('.pts', '') for i in files]
    return relpaths


def plot_kps_on_img(kps_oi, image, size=5):
    plt.imshow(kps_oi.draw_on_image(image, size=size))
    plt.show()


def crop_image_koi(image, koi, margin=0.20, border='linear_ramp'):
    x_axis_range = [koi.to_xy_array().T[0].min(), koi.to_xy_array().T[0].max()]
    x_axis_range = [(1 - margin) * x_axis_range[0], (1 + margin) * x_axis_range[1]]
    y_axis_range = [koi.to_xy_array().T[1].min(), koi.to_xy_array().T[1].max()]
    y_axis_range = [(1 - margin) * y_axis_range[0], (1 + margin) * y_axis_range[1]]

    x_axis_clipped = np.clip(x_axis_range, a_min=0, a_max=image.shape[1])
    y_axis_clipped = np.clip(y_axis_range, a_min=0, a_max=image.shape[0])

    img_cropped = image[round2int(y_axis_clipped[0]): round2int(y_axis_clipped[1]),
                  round2int(x_axis_clipped[0]): round2int(x_axis_clipped[1])]

    new_koi = koi.shift(x=-x_axis_range[0], y=-y_axis_range[0])

    # fix aspect ratio
    max_dim_len, ax_dim_fix = np.max(img_cropped.shape), np.argmin(img_cropped.shape[:2])
    dim_addition = (max_dim_len - img_cropped.shape[ax_dim_fix]) // 2
    if ax_dim_fix == 0:
        img_cropped = np.pad(img_cropped, pad_width=((dim_addition, dim_addition), (0, 0), (0, 0)), mode=border)
        new_koi = new_koi.shift(x=0, y=dim_addition)
    if ax_dim_fix == 1:
        img_cropped = np.pad(img_cropped, pad_width=((0, 0), (dim_addition, dim_addition), (0, 0)), mode=border)
        new_koi = new_koi.shift(x=dim_addition, y=0)

    # plot_kps_on_img(new_koi, img_cropped)

    return img_cropped, new_koi


def read_img(fname):
    try:
        img = imread(f'{fname}.png')
    except FileNotFoundError:
        img = imread(f'{fname}.jpg')
    return img


def points_bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]


def save_pair_img_pts(destination, basename, img_cropped, kps_io_cropped, debug=False):
    FileHandler.make_folder_if_not_exist_else_return_path(destination)
    FileHandler.make_folder_if_not_exist_else_return_path(os.path.dirname(os.path.join(destination, basename)))

    if debug:
        img_cropped = kps_io_cropped.draw_on_image(img_cropped, size=7)
    imsave(fname=f'{os.path.join(destination, basename)}.png', arr=img_cropped)
    # np.save(file=f'{os.path.join(destination, basename)}.npy', arr=kps_io_cropped.to_xy_array())
