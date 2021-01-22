# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from utils.file_handler import FileHandler
from main.refactor.transforms import fliplr_joints, crop, generate_target, transform_pixel
from utils.plot_utils import *


class DataSet68(data.Dataset):
    def __init__(self, df, workset_path, augmentation_args, model_args, is_train=True, transform=None):
        # specify annotation file for dataset
        self.df = df
        self.is_train = is_train
        self.transform = transform
        self.data_root = workset_path
        self.input_size = model_args['input_size']
        self.output_size = model_args['heatmap_size']
        self.sigma = float(model_args['sigma'])
        self.scale_factor = float(augmentation_args['scale'])
        self.rot_factor = augmentation_args['rotation_factor']
        self.label_type = model_args['target_type']
        self.flip = augmentation_args['flip_lr']

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_root, self.df.iloc[idx, 2], 'img', self.df.iloc[idx, 1]+'.jpg')
        pts_path = os.path.join(self.data_root, self.df.iloc[idx, 2], 'pts68', self.df.iloc[idx, 1]+'.pts')

        im_ = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        img = cv2.resize(im_, dsize=tuple(self.input_size), interpolation=cv2.INTER_CUBIC)
        scale = im_.shape[0] / img.shape[0]

        center_w = 0
        center_h = 0
        center = torch.Tensor([center_w, center_h])

        pts = np.array(FileHandler.load_json(pts_path)['pts'], dtype=np.float)

        scale *= 1.25   # TODO: why is this done?
        nparts = pts.shape[0]

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='300W')
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, rot=r)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                               scale, self.output_size, rot=r)
                target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                                            label_type=self.label_type)
        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts}

        plot_ldm_on_image(img, pts)
        plt.show()
        return img, target, meta

