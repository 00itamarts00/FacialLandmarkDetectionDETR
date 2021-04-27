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

        # iaa.Sometimes(0.2, iaa.Fliplr(1.0)),
        iaa.Sometimes(0.1, iaa.CropAndPad(percent=(-0.10, 0.10))),
        iaa.Sometimes(0.1, iaa.Affine(rotate=5)),

        # apply Gaussian blur with a sigma between 0 and 3 to 50% of the images
        # apply one of the augmentations: Dropout or CoarseDropout
        # iaa.OneOf([
        #    iaa.Sometimes(0.02, iaa.Dropout((0.01, 0.1), per_channel=0.5)),  # randomly remove up to 10% of the pixels
        #    iaa.Sometimes(0.02, iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2)),
        # ]),

        # apply from 0 to 3 of the augmentations from the list
        iaa.SomeOf((0, 3), [
            iaa.Sometimes(0.05, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.2))),  # sharpen images
            iaa.Sometimes(0.05, iaa.Emboss(alpha=(0, 1.0), strength=(0, 1.2))),  # emboss images
            iaa.Sometimes(0.05, iaa.GaussianBlur((0, 3.0))),
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
    ptsa = np.float64(np.asarray(ptsa))
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


def get_face68_flip():
    def vx(st, en=None, step=1):
        if en is None:
            return np.array(range(st, st + 1))

        exen = 1 if step > 0 else -1
        return np.array(range(st, en + exen, step))

    dl = list()
    dl.append([vx(1, 17), vx(17, 1, -1)])
    dl.append([vx(18, 22), vx(27, 23, -1)])
    dl.append([vx(23, 27), vx(22, 18, -1)]),
    dl.append([vx(28, 31), vx(28, 31)]),
    dl.append([vx(32, 36), vx(36, 32, -1)]),
    dl.append([vx(37, 40), vx(46, 43, -1)]),
    dl.append([vx(41), vx(48)]),
    dl.append([vx(42), vx(47)]),
    dl.append([vx(43, 46), vx(40, 37, -1)]),
    dl.append([vx(47), vx(42)]),
    dl.append([vx(48), vx(41)]),
    dl.append([vx(49, 55), vx(55, 49, -1)]),
    dl.append([vx(56, 60), vx(60, 56, -1)]),
    dl.append([vx(61, 65), vx(65, 61, -1)]),
    dl.append([vx(66, 68), vx(68, 66, -1)])

    sidx, didx = [], []
    for i in range(len(dl)):
        didx = didx + np.array(dl[i][0]).tolist()
        sidx = sidx + np.array(dl[i][1]).tolist()

    return np.asarray(sidx) - 1, np.asarray(didx) - 1


def fliplr(im, pts):
    ptsa = copy.deepcopy(pts)
    ima = np.fliplr(im)
    sidx, didx = get_face68_flip()

    ptsa[didx, 0] = pts[sidx, 0]
    ptsa[didx, 1] = pts[sidx, 1]
    ptsa[didx, 0] = ima.shape[1] - ptsa[didx, 0]
    return ima, ptsa


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
        self.gaurfactor = 5
        self.gaustd = 2.5
        # Extracted from trainset_full.csv
        self.mean = np.array([0.5021, 0.3964, 0.3471], dtype=np.float32)
        self.std = np.array([0.2858, 0.2547, 0.2488], dtype=np.float32)
        self.imga = create_base_gaussian(np.multiply(self.hmsize, self.gaurfactor), self.gaustd * self.gaurfactor)

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
        weighted_loss_mask_awing_bin = (dilation(rgb2gray(img), square(3)) / 256) >= 0.2
        weighted_loss_mask_awing = torch.Tensor(resize(weighted_loss_mask_awing_bin, self.hmsize))
        if self.transform is not None and self.is_train:
            if random.random() > 0.5:
                img, pts = fliplr(img, pts)
            img, pts = transform_data(self.transform, img, pts)

        heatmaps, hm_pts = create_heatmaps2(pts, np.shape(img), self.hmsize, self.imga, self.gaurfactor)
        heatmaps = np.float32(heatmaps)  # /np.max(hm)
        heatmaps = torch.Tensor(heatmaps)
        target = torch.Tensor(pts/255)

        img = (np.float32(img)/255 - self.mean) / self.std
        # img = np.float32(img) / 255
        img = torch.Tensor(img)
        img = img.permute(2, 0, 1)

        hmfactor = self.input_size[0] / self.hmsize[0]
        pts_ = torch.Tensor(pts_)

        item = {'index': idx, 'img_name': img_name, 'dataset': dataset,
                'img': img, 'heatmaps': heatmaps, 'hm_pts': hm_pts, 'opts': pts_, 'sfactor': sfactor,
                'hmfactor': hmfactor, 'tpts': pts, 'target': target,
                'weighted_loss_mask_awing': weighted_loss_mask_awing}
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
        img_ = 255 * (img_ * std + mean)
        img_[img_ > 255] = 255
        img_[img_ < 0] = 0

        return np.ubyte(img_)


if __name__ == '__main__':

    worksets_path = '../../worksets/WS02'
    datasets = ('WFLW/trainset', 'helen/trainset', 'lfpw/trainset', 'menpo/trainset', 'AFW')
    nickname = 'trainset_full'

    dflist = get_data_list(worksets_path, datasets, nickname)
    dftrain = dflist.sample(frac=0.8, random_state=42)  # random state is a seed value
    dfvalid = dflist.drop(dftrain.index)

    dftrain.to_csv(os.path.join(worksets_path, f'{nickname}_train.csv'))
    dfvalid.to_csv(os.path.join(worksets_path, f'{nickname}_valid.csv'))

    trainset = CLMDataset('../../worksets/WS02', dftrain, transform=get_def_transform())
    validset = CLMDataset('../../worksets/WS02', dfvalid)

    num_i = len(trainset)
    print(f'Number of valid images : {num_i}')

    num_i = len(validset)
    print(f'Number of train images : {num_i}')

    for i in range(5):
        item = trainset.__getitem__(i)
        imshowpts(np.array(item['img']).transpose([1, 2, 0]), item['pts'])
        imshowpts(np.array(item['hm']).transpose([1, 2, 0]).sum(2), item['hm_pts'])

    for i in range(5):
        item = validset.__getitem__(i)
        imshowpts(np.array(item['img']).transpose([1, 2, 0]), item['pts'])

    datasets = ('helen/testset', 'lfpw/testset', 'WFLW/testset', '300W', 'ibug', 'COFW68/COFW_test_color')
    nickname = 'testset_full'
    dftest = get_data_list(worksets_path, datasets, nickname)

    testset = CLMDataset('../../worksets/WS02', dftest)

    num_i = len(testset)
    print(f'Number of test images : {num_i}')

    for i in range(5):
        item = testset.__getitem__(i)
        imshowpts(np.array(item['img']).transpose([1, 2, 0]), item['pts'])

    '''
    if False :
        trainset  = CLMDataset('..\..\worksets\WS02',
                                ('WFLW/trainset','helen/trainset','lfpw/trainset','menpo/trainset','AFW'),
                                nickname = 'trainset_full')

        meanx,stdx= trainset.update_mean_and_std()
        print(f'meanx : {meanx}')
        print(f'stdx : {stdx}')
    '''
