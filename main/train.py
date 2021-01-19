from __future__ import print_function
import shutil
import json
from utils.file_handler import FileHandler
import os
from CLMDataset import CLMDataset, get_def_transform, get_data_list
from common.losslog import CLossLog
import pandas as pd
import torch
import wandb
import globals as g
from torch.utils import data
import math
from optimizer import OptimizerCLS
from schedule import ScheduleCLS
from models import model_LMDT01
from common.modelhandler import CModelHandler
from collections import namedtuple
from common.nnstats import CnnStats


class LDMTrain(object):
    def __init__(self, params):
        self.pr = params
        self.workset_path = os.path.join(self.ds['dataset_dir'], self.ds['workset_name'])
        self.paths = self.create_workspace()

    @property
    def ds(self): return self.pr['dataset']
    @property
    def tr(self): return self.pr['train']
    @property
    def ex(self): return self.pr['experiment']

    def create_training_data(self):
        datasets = self.tr['datasets']['to_use']
        trainset_partition = self.tr['trainset_partition']
        partition_seed = self.tr['partition_seed']
        use_augmentations = self.tr['datasets']['use_augmentations']
        nickname = 'trainset_full'

        df = get_data_list(worksets_path=self.workset_path, datasets=datasets, nickname=nickname, numpts=68)
        dftrain = df.sample(frac=trainset_partition, random_state=partition_seed)  # random state is a seed value
        dfvalid = df.drop(dftrain.index)

        dftrain.to_csv(os.path.join(self.workset_path, f'{nickname}_train.csv'))
        dfvalid.to_csv(os.path.join(self.workset_path, f'{nickname}_valid.csv'))

        transform = get_def_transform() if use_augmentations else None

        trainset = CLMDataset(self.workset_path, dftrain, transform=transform)
        validset = CLMDataset(self.workset_path, dfvalid)

        return trainset, validset

    def create_workspace(self):
        os.makedirs(self.ex['workspace_path'], exist_ok=True)
        workspace_path = os.path.join(self.ex['workspace_path'], self.ex['name'], g.TIMESTAMP)
        structure = {'workspace': workspace_path,
                     'nets': os.path.join(workspace_path, 'nets'),
                     'args': os.path.join(workspace_path, 'args'),
                     'logs': os.path.join(workspace_path, 'logs'),
                     'stats': os.path.join(workspace_path, 'stats'),
                     'workset': self.workset_path
                     }
        paths = FileHandler.dict_to_nested_namedtuple(structure)
        [os.makedirs(i, exist_ok=True) for i in paths]
        FileHandler.save_dict_as_yaml(self.pr, os.path.join(workspace_path, 'params.yaml'))
        return paths

    def load_data_to_dataloader(self, trainset, validset, **kwargs):
        batch_size = self.tr['batch_size']
        train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
        valid_loader = data.DataLoader(validset, batch_size=batch_size, shuffle=False, **kwargs)
        return train_loader, valid_loader

    def load_losslog(self):
        losslog = CLossLog(self.paths)
        return losslog

    def load_optimizer(self, model):
        opt = OptimizerCLS(params=self.pr, model=model)
        return opt.load_optimizer()

    def load_model(self):
        kwargs = {'workspace_path': self.paths.workspace, 'epochs_to_save': None}
        if self.pr['model']['name'] == 'LMDT01':
            output_branch = self.pr['model']['output_branch']
            model = model_LMDT01.get_instance(output_branch=output_branch)
        mdhl = CModelHandler(model, **kwargs)
        model, last_epoch = mdhl.load()
        return model, last_epoch

    def load_scheduler(self, optimizer):
        sc = ScheduleCLS(params=self.pr, optimizer=optimizer)
        return sc.load_scheduler()

    def backend_operations(self):
        cuda = self.tr['cuda']
        torch.manual_seed(self.tr['torch_seed'])
        use_cuda = cuda['use'] and torch.cuda.is_available()
        device = torch.device(cuda['device_type'] if use_cuda else 'cpu')
        torch.backends.benchmark = self.tr['backend']['use_torch']
        return device


    def train(self):
        device = self.backend_operations()
        trainset, validset = self.create_training_data()
        train_loader, valid_loader = self.load_data_to_dataloader(trainset, validset)
        model, last_epoch = self.load_model()
        optimizer = self.load_optimizer(model=model)
        losslog = self.load_losslog()
        scheduler = self.load_scheduler(optimizer)
        meta_model = losslog.load(last_epoch)
        trn_loss = meta_model['params']['loss_train'][-1][1] if meta_model is not None else 0
        nnstats = CnnStats(self.paths.stats, model)
        model.to(device)

        # model, trn_loss, vld_loss = train_model(device, train_loader, valid_loader, model, workspace_path, config,
        #                                         logs_path=logs_path, do_valid=do_valid)



