from __future__ import print_function

import copy
import math
import os
# import shutil
# import json
import time

from torch.utils import data
from main.refactor.utils import save_checkpoint
from models import hrnet_config
from models.hrnet_config import update_config
from tensorboardX import SummaryWriter
from main.refactor.functions import train, validate, inference
# import pandas as pd
# import torch
# import wandb
from common.losslog import CLossLog
from common.modelhandler import CModelHandler
from common.nnstats import CnnStats
from main.components.CLMDataset import CLMDataset, get_def_transform, get_data_list
from main.components.optimizer import OptimizerCLS
# import numpy as np
from main.components.evaluate_model import *
from main.components.scheduler_cls import ScheduleCLS
from utils.file_handler import FileHandler
from models import model_LMDT01, HRNET
from main.refactor.dataset import DataSet68
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
        self.loss = self.load_criteria()
        self.writer = self.init_writer()

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

    def init_writer(self):
        writer_dict = {
            'writer': SummaryWriter(log_dir=self.paths.logs),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }
        return writer_dict

    def load_criteria(self):
        loss_crit = self.tr['criteria']
        args = self.pr['loss'][loss_crit]
        if loss_crit == 'MSELoss':
            return torch.nn.MSELoss(size_average=True)

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

        # dftrain.to_csv(os.path.join(self.workset_path, f'{nickname}_train.csv'))
        # dfvalid.to_csv(os.path.join(self.workset_path, f'{nickname}_valid.csv'))

        transform = get_def_transform() if use_augmentations else None

        # trainset = CLMDataset(self.workset_path, dftrain, transform=transform)
        # validset = CLMDataset(self.workset_path, dfvalid)

        augmentation_args = None
        if self.tr['datasets']['use_augmentations']:
            augmentation_args = self.tr['datasets']['augmentations']
        model_name = self.tr['model']
        model_args = self.pr['model'][model_name]

        trainset = DataSet68(dftrain, self.paths.workset, augmentation_args, model_args,
                             is_train=True, transform=None)
        validset = DataSet68(dfvalid, self.paths.workset, augmentation_args, model_args,
                             is_train=False, transform=None)

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
                     'final_output_dir': os.path.join(workspace_path, 'final_output_dir'),
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
        if self.tr['model'] == 'HRNET':
            config = hrnet_config._C
            model = HRNET.get_face_alignment_net(config)
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
            args = [output, target, opts]
            kwargs = {}
            loss = self.mdhl.model.loss(args, **kwargs)

            vmean_loss = []
            if len(loss) > 1:
                for lidx in range(0, len(loss) - 1):
                    loss[lidx].backward(retain_graph=True)
                    vmean_loss.append(np.mean(loss[lidx].item()))
                loss[-1].backward()
                mean_loss = np.mean(loss[-1].item())
                vmean_loss.append(mean_loss)
            else:
                loss[0].backward()
                mean_loss = np.mean(loss[0].item())
                vmean_loss.append(mean_loss)

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
                args = [output, target, opts]
                kwargs = {}
                loss = self.mdhl.model.loss(args, **kwargs)
                mean_loss = np.mean(loss[-1].item())
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

        # TODO: support multiple gpus
        # gpus = list(config.GPUS)
        # model = torch.nn.DataParallel(self.mdhl.model, device_ids=gpus).cuda()

        epochs = self.tr['epochs'] + self.mdhl.last_epoch + 1
        best_nme = 100
        nme = 0
        predictions = None
        for epoch in range(self.mdhl.last_epoch + 1, epochs):
            if math.isnan(self.trn_loss) or math.isinf(self.trn_loss):
                break
            if self.train_loader is not None:
                starttime = time.time()
                # train
                kwargs = {'log_interval': 20}
                train(train_loader=self.train_loader,
                      model=self.mdhl.model,
                      criterion=self.loss,
                      optimizer=self.optimizer,
                      epoch=epoch,
                      writer_dict=self.writer,
                      **kwargs)

                # evaluate
                kwargs = {'num_landmarks': self.tr['num_landmarks']}
                nme, predictions = validate(val_loader=self.valid_loader,
                                            model=self.mdhl.model,
                                            critertion=self.loss,
                                            epoch=epoch,
                                            writer_dict=self.writer,
                                            **kwargs)
            self.scheduler.step()

            is_best = nme < best_nme
            best_nme = min(nme, best_nme)
            logger.info(f'=> saving checkpoint to {self.paths.final_output_dir}')
            final_model_state_file = os.path.join(self.paths.final_output_dir, 'final_state.pth')

            save_checkpoint(states=
                            {"state_dict": self.mdhl.model,
                             "epoch": epoch + 1,
                             "best_nme": best_nme,
                             "optimizer": self.optimizer.state_dict()},
                            predictions=predictions,
                            is_best=is_best,
                            output_dir=self.paths.final_output_dir,
                            filename='checkpoint_{}.pth'.format(epoch))

            logger.info(f'saving final model state to {final_model_state_file}')
            torch.save(self.mdhl.model.state_dict(), final_model_state_file)
            self.writer['writer'].close()
