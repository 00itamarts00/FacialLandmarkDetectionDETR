from __future__ import print_function

import copy
import logging
import math
import os
# import shutil
# import json
import time

from torch.utils import data

# import pandas as pd
# import torch
# import wandb
import main.globals as g
from common.losslog import CLossLog
from common.modelhandler import CModelHandler
from common.nnstats import CnnStats
from main.CLMDataset import CLMDataset, get_def_transform, get_data_list
from main.components.optimizer import OptimizerCLS
# import numpy as np
from main.evaluate_model import *
from models import model_LMDT01
from main.components.scheduler_cls import ScheduleCLS
from utils.file_handler import FileHandler

torch.cuda.empty_cache()
logger = logging.getLogger(__name__)


class LDMTrain(object):
    def __init__(self, params):
        self.pr = params
        self.workset_path = os.path.join(self.ds['dataset_dir'], self.ds['workset_name'])
        self.paths = self.create_workspace()
        self.device = self.backend_operations()
        self.train_loader, self.valid_loader = self.create_dataloaders()
        self.mdhl = self.load_model()
        self.optimizer = self.load_optimizer()
        self.losslog = self.load_losslog()
        self.trn_loss = self.get_last_loss()
        self.scheduler = self.load_scheduler()
        self.nnstats = CnnStats(self.paths.stats, self.mdhl.model)

    @property
    def hm_amp_factor(self):
        return self.tr['hm_amp_factor']

    @property
    def log_interval(self):
        return self.ex['log_interval']

    @property
    def ds(self):
        return self.pr['dataset']

    @property
    def tr(self):
        return self.pr['train']

    @property
    def ex(self):
        return self.pr['experiment']

    @staticmethod
    def _get_data_from_item(item):
        sample = copy.deepcopy(item['img'])
        target = copy.deepcopy(item['hm'])
        opts = copy.deepcopy(item['opts'])
        sfactor = item['sfactor']
        for i in range(len(sfactor)):
            opts[i] = np.multiply(opts[i], sfactor[i]) / sample.size(2)
        return sample, target, opts

    def get_last_loss(self, type='train'):
        if self.losslog.meta_model == dict():
            return 0
        return self.losslog.meta_model[next(reversed(self.losslog.meta_model))][f'loss_{type}']['val']

    def create_dataloaders(self):
        use_cuda = self.tr['cuda']['use']
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

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        batch_size = self.tr['batch_size']
        train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
        valid_loader = data.DataLoader(validset, batch_size=batch_size, shuffle=False, **kwargs)

        return train_loader, valid_loader

    def create_workspace(self):
        workspace_path = self.pr['workspace_path']
        structure = {'workspace': workspace_path,
                     'nets': os.path.join(workspace_path, 'nets'),
                     'args': os.path.join(workspace_path, 'args'),
                     'logs': os.path.join(workspace_path, 'logs'),
                     'stats': os.path.join(workspace_path, 'stats'),
                     'workset': self.workset_path
                     }
        paths = FileHandler.dict_to_nested_namedtuple(structure)
        [os.makedirs(i, exist_ok=True) for i in paths]
        return paths

    def load_losslog(self):
        losslog = CLossLog(self.paths, last_epoch=self.mdhl.last_epoch)
        return losslog

    def load_optimizer(self):
        opt = OptimizerCLS(params=self.pr, model=self.mdhl.model)
        return opt.load_optimizer()

    def load_model(self):
        model = None
        kwargs = {'workspace_path': self.paths.workspace, 'epochs_to_save': None}
        if self.tr['model'] == 'LMDT01':
            output_branch = self.pr['model']['LMDT01']['output_branch']
            model = model_LMDT01.get_instance(output_branch=output_branch)
        mdhl = CModelHandler(model=model, nets=self.paths.nets, args=self.paths.args, **kwargs)
        return mdhl

    def load_scheduler(self):
        sc = ScheduleCLS(params=self.pr, optimizer=self.optimizer, last_epoch=self.mdhl.last_epoch)
        return sc.load_scheduler()

    def backend_operations(self):
        cuda = self.tr['cuda']
        torch.manual_seed(self.tr['torch_seed'])
        use_cuda = cuda['use'] and torch.cuda.is_available()
        device = torch.device(cuda['device_type'] if use_cuda else 'cpu')
        torch.backends.benchmark = self.tr['backend']['use_torch']
        return device

    def train_epoch(self, epoch):
        self.mdhl.model.train()
        train_loss = []
        for batch_idx, item in enumerate(self.train_loader):
            sample, target, opts = self._get_data_from_item(item)
            target = np.multiply(target, self.hm_amp_factor)
            sample, target, opts = sample.to(self.device), target.to(self.device), opts.to(self.device)
            self.optimizer.zero_grad()
            output = self.mdhl.model(sample)
            loss = self.mdhl.model.loss(output, target, opts)
            # TODO: understand better the way be backprop the loss
            vmean_loss = []
            if len(loss) > 1:
                for lidx in range(0, len(loss) - 1):
                    loss[lidx].backward(retain_graph=True)
                    vmean_loss.append(np.mean(loss[lidx].item()))
                loss[-1].backward()
                mean_loss = np.mean(loss[-1].item())
                vmean_loss.append(mean_loss)
            else:
                loss.backward()
                mean_loss = np.mean(loss.item())
                vmean_loss[0] = mean_loss

            self.optimizer.step()

            train_loss.append(mean_loss)
            if batch_idx % self.log_interval == 0:
                bsz, ssz, per = (batch_idx + 1) * len(sample), len(self.train_loader.dataset), \
                                100. * (batch_idx + 1) / len(self.train_loader)
                mean_loss = np.mean(train_loss)
                osum0 = np.array(output[0].cpu().detach()).sum()
                osum3 = np.array(output[3].cpu().detach()).sum()
                print(f'Train Epoch: {epoch} [{bsz} / {ssz} ({per:.02f}%)]\tLoss: {mean_loss:.09f} \tosum0:'
                      f' {osum0:.02f}\tosum3: {osum3:.02f} -- {vmean_loss}')
            if self.ex['single_batch_debug']:
                break
        return np.mean(train_loss)

    def update_tensorboard(self, epoch, **kwargs):
        for key, val in kwargs.items():
            self.losslog.add_value(epoch, key, val)

    def valid_epoch(self):
        self.mdhl.model.eval()
        valid_loss = []
        epts_batch = dict()
        with torch.no_grad():
            for batch_idx, item in enumerate(self.valid_loader):
                sample, target, opts = self._get_data_from_item(item)
                target = np.multiply(target, self.hm_amp_factor)
                sample, target, opts = sample.to(self.device), target.to(self.device), opts.to(self.device)
                output = self.mdhl.model(sample)
                loss = self.mdhl.model.loss(output, target, opts)
                if len(loss) > 1:
                    mean_loss = np.mean(loss[-1].item())
                else:
                    mean_loss = np.mean(loss.item())
                valid_loss.append(mean_loss)

                epts = self.mdhl.model.extract_epts(output, res_factor=1)
                epts = add_metadata_to_result(epts, item)
                epts_batch.update(epts)

                if batch_idx % self.log_interval == 0:
                    print(f'Validation:  [{(batch_idx + 1) * len(sample)}/{len(self.valid_loader.dataset)}'
                          f' ({100. * (batch_idx + 1) / len(self.valid_loader):.02f}%)]'
                          f'\tLoss: {np.mean(valid_loss):.06f}')
                if self.ex['single_batch_debug']:
                    break
        auc08, nle, fail08, bins, ced68 = calc_accuarcy(epts_batch)
        # auc08, nle = -1, -1
        return np.mean(valid_loss), auc08, nle

    def train(self):
        run_valid = self.tr['run_valid']
        self.mdhl.model.to(self.device)
        epochs = self.tr['epochs'] + self.mdhl.last_epoch + 1

        for epoch in range(self.mdhl.last_epoch + 1, epochs):
            if math.isnan(self.trn_loss) or math.isinf(self.trn_loss):
                break
            if self.train_loader is not None:
                starttime = time.time()
                trn_loss = self.train_epoch(epoch=epoch)
                runtime = (time.time() - starttime) / len(self.train_loader.dataset)
                last_lr = self.scheduler.get_last_lr()[0]

                print(f'Train set: Average loss: {trn_loss:.6f} LR={last_lr}\n')
                res = {'loss_train': trn_loss, 'lr': last_lr, 'runtime_train': runtime}
                self.update_tensorboard(epoch, **res)
                self.nnstats.add_measure(epoch, self.mdhl.model, dump=True)

            if run_valid and self.valid_loader is not None:
                starttime = time.time()
                vld_loss, auc08, nle = self.valid_epoch()
                runtime = (time.time() - starttime) / len(self.valid_loader.dataset)
                res = {'loss_valid': vld_loss, 'auc08_valid': auc08, 'nle_valid': nle, 'runtime_valid': runtime}
                self.update_tensorboard(epoch, **res)
                print(f'Valid set: Average loss: {vld_loss:.6f} auc08={auc08:.03f} nle={nle:.03f}, ''\n')
            self.scheduler.step()

            if self.ex['save_model']:
                self.mdhl.save(self.mdhl.model, epoch)
                self.losslog.dump()

        model, last_epoch = self.mdhl.model, self.mdhl.last_epoch

        trn_loss = self.get_last_loss(type='train')
        vld_loss = self.get_last_loss(type='valid')

        return model, trn_loss, vld_loss
