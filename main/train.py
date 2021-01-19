from __future__ import print_function

import json
import os
from CLMDataset import CLMDataset, get_def_transform, get_data_list
import pandas as pd
import torch
import wandb


class LDMTrain(object):
    def __init__(self, params):
        self.pr = params
        self.workset_path = os.path.join(self.ds['dataset_dir'], self.ds['workset_name'])

    @property
    def ds(self): return self.pr['dataset']
    @property
    def tr(self): return self.pr['train']

    def train(self):
        datasets = self.tr['datasets']['to_use']
        trainset_partition = self.tr['trainset_partition']
        partition_seed = self.tr['partition_seed']
        use_augmentations = self.tr['datasets']['use_augmentations']
        cuda = self.tr['cuda']
        nickname = 'trainset_full'

        df = get_data_list(worksets_path=self.workset_path, datasets=datasets, nickname=nickname, numpts=68)
        dftrain = df.sample(frac=trainset_partition, random_state=partition_seed)  # random state is a seed value
        dfvalid = df.drop(dftrain.index)

        dftrain.to_csv(os.path.join(self.workset_path, f'{nickname}_train.csv'))
        dfvalid.to_csv(os.path.join(self.workset_path, f'{nickname}_valid.csv'))

        transform = get_def_transform() if use_augmentations else None

        trainset = CLMDataset(self.workset_path, dftrain, transform=transform)
        validset = CLMDataset(self.workset_path, dfvalid)

        pass
