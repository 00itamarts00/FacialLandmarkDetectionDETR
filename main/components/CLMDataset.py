import os
import random
from copy import copy

import imgaug as ia
import imgaug.augmenters as iaa
import menpo.io as mio
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from skimage.morphology import dilation, square
from tqdm import tqdm

# from torchvision.utils.transforms import fliplr_joints, crop, generate_target, transform_pixel
import common.fileutils as fu
from common.ptsutils import create_heatmaps2
from create_dataset_utils.txt_parser import get_pts_from_txt
from main.components.ptsutils import fliplr_img_pts, gray2rgb_, normalize_meanstd, ch_last, ch_first
from main.globals import IMG_SUFFIX, BBS_DB


class CLMDataset(data.Dataset):
    def __init__(self, params, paths, dflist, is_train=True, transform=None):
        self.worksets_path = paths.workset
        self.transform = transform
        self.num_landmarks = params.train.num_landmarks
        self.input_size = params.train.input_size
        self.dflist = dflist
        self.is_train = is_train
        self.heatmaps = params.train.heatmaps if params.train.heatmaps.train_with_heatmaps else None
        # Extracted from trainset_full.csv
        # self.mean = np.array([0.52907253, 0.40422649, 0.34813007], dtype=np.float32)
        # self.std = np.array([0.2799659, 0.24529966, 0.24069724], dtype=np.float32)

    def __len__(self):
        return len(self.dflist)

    @staticmethod
    def get_img_path(basename_path):
        for suffix in IMG_SUFFIX:
            fname = f'{basename_path}.{suffix}'
            if os.path.exists(fname):
                return fname
        return None

    def get_img_name(self, idx):
        basename_path = self.dflist.iloc[idx]['basename_path']
        img_path = self.get_img_path(basename_path)
        dataset = self.dflist.iloc[idx]['dataset']
        return dataset, img_path

    @staticmethod
    def get_bbs_of_face(img_path, dataset):
        bb_pts_fname = os.path.join(BBS_DB, dataset, f'{os.path.splitext(os.path.basename(img_path))[0]}.pts')
        return get_pts_from_txt(bb_pts_fname)

    @staticmethod
    def crop_image_from_center_bb_mdm(mio_item, bb_pts, boundary_margin=1.3):
        center_crop = np.array(bb_pts).mean(0)
        range_crop = np.array(bb_pts).max(0) - np.array(bb_pts).min(0)
        min_indices = np.round(center_crop - range_crop.max() * boundary_margin / 2).astype(int)
        max_indices = np.round(center_crop + range_crop.max() * boundary_margin / 2).astype(int)
        mio_item, transform = mio_item.crop(min_indices=min_indices[::-1], max_indices=max_indices[::-1],
                                            constrain_to_boundary=True, return_transform=True)
        return mio_item, transform

    def __getitem__(self, idx):
        dataset, img_path = self.get_img_name(idx)
        mio_item = mio.import_image(filepath=img_path)
        opts = mio_item.landmarks.get('PTS').points
        bb_pts = self.get_bbs_of_face(img_path=img_path, dataset=dataset)
        mio_item, transform = self.crop_image_from_center_bb_mdm(mio_item, bb_pts)
        mio_item = mio_item.resize(self.input_size)
        # mio_item.view_landmarks(render_numbering=True, render_lines=True, render_axes=True, marker_style='.')
        imgo = ch_last(gray2rgb_(mio_item.pixels) * 256).astype(np.uint8)

        pts = mio_item.landmarks.get('PTS').points[:, ::-1]
        img = ch_first(gray2rgb_(normalize_meanstd(mio_item.pixels)))
        # img = (np.float32(imgo) - self.mean.reshape(-1, 1, 1)) / self.std.reshape(-1, 1, 1)

        if self.transform is not None and self.is_train:
            if random.random() > 0.5:
                img, pts = fliplr_img_pts(img, pts)
            img, pts = transform_data(self.transform, img, pts)

        img = torch.Tensor(img)
        pts = torch.Tensor(copy(pts))

        if self.heatmaps is not None:
            heatmaps, hm_pts = create_heatmaps2(pts, np.shape(img), self.heatmaps.heatmap_size,
                                                self.heatmaps.guassian_std)
            heatmaps = np.float32(heatmaps)
            hm_sum = np.sum(heatmaps, axis=0)

            heatmaps = torch.Tensor(heatmaps)
            # see: https://arxiv.org/pdf/1904.07399v3.pdf
            weighted_loss_mask_awing = dilation(hm_sum, square(3)) >= 0.2
            hmfactor = self.input_size[0] / self.heatmaps.heatmap_size[0]

        item = {'index': idx, 'img_name': mio_item.path.name, 'imgo': imgo,
                'dataset': dataset, 'img': img, 'opts': opts, 'tpts': pts}

        if self.heatmaps is not None:
            item.update({'heatmaps': heatmaps, 'hm_pts': hm_pts, 'hmfactor': hmfactor,
                         'weighted_loss_mask_awing': weighted_loss_mask_awing})
        return item

    def update_mean_and_std(self):
        n = self.__len__()
        dims = self.__getitem__(0)['img'].shape

        x = x2 = 0
        for i in tqdm(range(n)):
            item = self.__getitem__(i)
            x += np.sum(item['imgo'], axis=(1, 2))
            x2 += np.sum(np.power(item['imgo'], 2), axis=(1, 2))
        meanx = x / (n * dims[1] * dims[2])
        stdx = ((x2 - (n * dims[1] * dims[2]) * meanx ** 2) / (n * dims[1] * dims[2])) ** 0.5

        self.mean, self.std = meanx, stdx

        return meanx, stdx

    def renorm_image(self, img):
        img_ = np.array(img).transpose([1, 2, 0])
        img_ = 256 * (img_ * self.std + self.mean)
        img_ = np.clip(img_, a_min=0, a_max=255)
        return np.ubyte(img_)


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
    try:
        image_aug, kps_aug = np.array(transform(image=im, keypoints=kps), dtype=object)
    except AssertionError:
        print('here')
    ptsa = []
    for item in kps_aug:
        ptsa.append([item.coords[0][0], item.coords[0][1]])
    ptsa = np.float32(np.asarray(ptsa))
    ima = image_aug
    return ima, ptsa


def get_data_list(worksets_path, datasets, nickname):
    csv_file = os.path.join(worksets_path, f'{nickname}.csv')
    dflist = pd.DataFrame()
    for dataset in datasets:
        df = pd.DataFrame()
        ds_path = os.path.join(worksets_path, dataset)
        ptsfilelist = fu.get_files_list(ds_path, ('.pts'))
        base_names = [os.path.splitext(f)[0] for f in ptsfilelist]
        df['basename_path'] = base_names
        df['dataset'] = dataset
        dflist = pd.concat([dflist, df], ignore_index=True)
    dflist.to_csv(csv_file)
    return dflist
