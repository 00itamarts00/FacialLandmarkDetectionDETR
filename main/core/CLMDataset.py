# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import copy
import json
import os
import random
from PIL import Image

from main.components.ptsutils import fliplr_img_pts
from utils.file_handler import FileHandler
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from skimage.morphology import dilation, square
from skimage.color import rgb2gray
from skimage.transform import resize
# from torchvision.utils.transforms import fliplr_joints, crop, generate_target, transform_pixel
import common.fileutils as fu
from common.ptsutils import imshowpts, create_heatmaps2, create_base_gaussian


def get_def_transform():
    ia.seed(random.randint(0, 1000))

    aug_pipeline = iaa.Sequential([
        # iaa.Sometimes(0.1, iaa.Multiply((0.9, 1.2))), # change brightness, doesn't affect keypoints
        iaa.Sometimes(0.1, iaa.CropAndPad(percent=(-0.10, 0.10))),
        iaa.Sometimes(0.1, iaa.Affine(rotate=(-30, 30))),
        iaa.Sometimes(0.1, iaa.Affine(scale=(0.75, 1.25))),
        iaa.Sometimes(0.1, iaa.contrast.LinearContrast(alpha=(0.6, 1.4))),

        # apply from 0 to 3 of the augmentations from the list
        iaa.SomeOf((0, 3), [
            iaa.Sometimes(0.1, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.2))),  # sharpen images
            iaa.Sometimes(0.1, iaa.Emboss(alpha=(0, 1.0), strength=(0, 1.2))),  # emboss images
            iaa.Sometimes(0.1, iaa.GaussianBlur((0, 2.0))),
        ])
    ],
        random_order=True  # apply the augmentations in random order
    )
    return aug_pipeline


def transform_data(transform, im, pts):
    kps = [ia.Keypoint(x, y) for x, y in pts]
    image_aug, kps_aug = np.array(transform(image=im, keypoints=kps), dtype=object)
    ptsa = []
    for item in kps_aug:
        ptsa.append([item.coords[0][0], item.coords[0][1]])
    ptsa = np.float32(np.asarray(ptsa))
    ima = image_aug
    return ima, ptsa


def get_data_list(worksets_path, datasets, nickname, numpts=68):
    csvfile = os.path.join(worksets_path, f'{nickname}.csv')
    print(os.getcwd())
    if os.path.exists(csvfile):
        dflist = pd.read_csv(csvfile)
    else:
        dflist = pd.DataFrame()
        for dataset in datasets:
            df = pd.DataFrame()
            ptsdir = os.path.join(worksets_path, dataset, f'pts{numpts}')
            if os.path.exists(ptsdir):
                ptspath = os.path.join(worksets_path, dataset, f'pts{numpts}')
                ptsfilelist = fu.get_files_list(ptspath, ('.pts'))
                imgnames = [os.path.splitext(os.path.relpath(f, ptspath))[0] for f in ptsfilelist]
                df['imgnames'] = imgnames
                df['dataset'] = dataset
                dflist = pd.concat([dflist, df], ignore_index=True)
        dflist.to_csv(csvfile)
    return dflist


class CLMDataset(data.Dataset):
    def __init__(self, params, paths, dflist, is_train=True, transform=None):
        self.worksets_path = paths.workset
        self.transform = transform
        self.num_landmarks = params['train']['num_landmarks']
        model_args = params['model'][params['train']['model']]
        self.dflist = dflist
        self.is_train = is_train
        self.input_size = model_args['input_size']
        self.hmsize = model_args['heatmap_size']
        self.gaustd = 1.5
        # Extracted from trainset_full.csv
        self.mean = np.array([0.5021, 0.3964, 0.3471], dtype=np.float32)
        self.std = np.array([0.2858, 0.2547, 0.2488], dtype=np.float32)

    def __len__(self):
        return len(self.dflist)

    def get_pairdata(self, idx):
        df = self.dflist
        imgname = df.iloc[idx]['imgnames']
        dataset = df.iloc[idx]['dataset']

        pts_path = os.path.join(self.worksets_path, dataset, f'pts{self.num_landmarks}', f'{imgname}.pts')
        img_path = os.path.join(self.worksets_path, dataset, 'img', f'{imgname}.jpg')

        im_ = np.array(Image.open(img_path), dtype=np.float32)
        pts_ = np.array(FileHandler.load_json(pts_path)['pts'])
        return im_, pts_

    def get_infodata(self, idx):
        imgname = self.dflist.iloc[idx]['imgnames']
        dataset = self.dflist.iloc[idx]['dataset']
        return dataset, imgname

    def __getitem__(self, idx):
        dataset, img_name = self.get_infodata(idx)
        im_, pts_ = self.get_pairdata(idx)

        img = ia.imresize_single_image(im_, self.input_size)
        sfactor = img.shape[0] / im_.shape[0]
        pts = pts_ * sfactor
        if self.transform is not None and self.is_train:
            if random.random() > 0.5:
                img, pts = fliplr_img_pts(img, pts)  # dataset=dataset.split('/')[0].upper())
            img, pts = transform_data(self.transform, img, pts)

        heatmaps, hm_pts = create_heatmaps2(pts, np.shape(img), self.hmsize, self.gaustd)
        heatmaps = np.float32(heatmaps)  # /np.max(hm)
        hm_sum = np.sum(heatmaps, axis=0)

        heatmaps = torch.Tensor(heatmaps)
        # see: https://arxiv.org/pdf/1904.07399v3.pdf
        weighted_loss_mask_awing = dilation(hm_sum, square(3)) >= 0.2

        img = (np.float32(img) / 256 - self.mean) / self.std
        img = torch.Tensor(img)
        img = img.permute(2, 0, 1)

        hmfactor = self.input_size[0] / self.hmsize[0]
        pts_ = torch.Tensor(pts_)

        item = {'index': idx, 'img_name': img_name, 'dataset': dataset,
                'img': img, 'heatmaps': heatmaps, 'hm_pts': hm_pts, 'opts': pts_, 'sfactor': sfactor,
                'hmfactor': hmfactor, 'tpts': pts, 'weighted_loss_mask_awing': weighted_loss_mask_awing}
        return item

    def update_mean_and_std(self):
        n = self.__len__()
        dims = np.array(self.__getitem__(0)['img']).shape

        x = 0
        x2 = 0
        for i in range(n):
            item = self.__getitem__(i)
            im = item['img']
            x = x + torch.sum(im, [1, 2])
            x2 = x2 + torch.sum(torch.pow(im, 2), [1, 2])

            if not i % 100:
                print(f'{i} out of {n}')

        meanx = x / (n * dims[1] * dims[2])
        stdx = ((x2 - (n * dims[1] * dims[2]) * meanx ** 2) / (n * dims[1] * dims[2])) ** 0.5

        self.mean = meanx
        self.std = stdx

        return meanx, stdx

    def renorm_image(self, img):
        mean = np.array([0.5021, 0.3964, 0.3471], dtype=np.float32)
        std = np.array([0.2858, 0.2547, 0.2488], dtype=np.float32)

        img_ = np.array(img).transpose([1, 2, 0])
        img_ = 256 * (img_ * std + mean)
        img_ = np.clip(img_, a_min=0, a_max=255)

        return np.ubyte(img_)
