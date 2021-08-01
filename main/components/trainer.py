from __future__ import print_function

import logging
import math
import os
import sys

import pandas as pd
import torch.backends.cudnn
import torch.optim as optim
from clearml.logger import Logger
from dotmap import DotMap
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils import data
import main.globals as g
from main.components.CLMDataset import CLMDataset, get_def_transform, get_data_list
from main.core.functions import train_epoch, validate_epoch
from main.core.nnstats import CnnStats
from main.core.utils import save_checkpoint
from main.detr.models.detr import build as build_model
from main.detr.models.detr import load_criteria as load_criteria_detr
from models.HRNET import hrnet_config, update_config
from models.HRNET.HRNET import get_face_alignment_net
from models.HRNET.hrnet_utils import get_optimizer

torch.cuda.empty_cache()


# TODO: Load tensorboard logs as df/dict


class LDMTrain(object):
    def __init__(self, params, single_image_train=False, last_epoch=0, logger=None):
        self.params = params
        self.logger_cml = logger
        self.single_image_train = single_image_train
        self.workset_path = os.path.join(self.ds.dataset_dir, self.ds.workset_name)
        self.paths = self.create_workspace()
        self.last_epoch = last_epoch
        self.device = self.backend_operations()
        self.train_loader, self.valid_loader = self.create_dataloaders()
        self.model = self.load_model()
        self.criteria = self.load_criteria()
        self.optimizer = self.load_optimizer()
        self.scheduler = self.load_scheduler()
        self.nnstats = CnnStats(self.paths.stats, self.model)
        self.trn_loss = 0

    def load_criteria(self):
        if self.tr.model == 'DETR':
            return load_criteria_detr(args=self.pr.detr_args)
        if self.tr.model == 'HRNET':
            return torch.nn.MSELoss(size_average=True).cuda()

    @property
    def pr(self):
        return self.params

    @property
    def ds(self):
        return self.pr.dataset

    @property
    def tr(self):
        return self.pr.train

    def create_dataloaders(self):
        nickname = 'trainset_full'

        df = get_data_list(worksets_path=self.workset_path, datasets=self.tr.datasets.to_use, nickname=nickname,
                           numpts=68)
        dftrain = df.sample(frac=self.tr.trainset_partition, random_state=self.tr.partition_seed)
        # random state is a seed value
        # dfvalid = df.drop(dftrain.index)
        dfvalid = pd.read_csv('/home/itamar/thesis/DATASET/WS02/300W.csv')
        transform = get_def_transform() if self.tr.datasets.use_augmentations else None

        trainset = CLMDataset(self.pr, self.paths, dftrain, transform=transform)
        validset = CLMDataset(self.pr, self.paths, dfvalid)

        num_workers = self.tr.cuda.num_workers if sys.gettrace() is None else 0
        kwargs = {'num_workers': num_workers, 'pin_memory': self.tr.cuda.pin_memory} if self.tr.cuda.use else {}

        batch_size = self.tr.batch_size if not self.single_image_train else 1
        train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
        valid_loader = data.DataLoader(validset, batch_size=batch_size, shuffle=False, **kwargs)
        self.logger_cml.report_text(f'Number of train images : {len(trainset)}', level=logging.INFO, print_console=True)
        self.logger_cml.report_text(f'Number of valid images : {len(validset)}', level=logging.INFO, print_console=True)
        return train_loader, valid_loader

    def create_workspace(self):
        workspace_path = self.pr.workspace_path
        structure = {
            "workspace": workspace_path,
            "checkpoint": os.path.join(workspace_path, "checkpoint"),
            "args": os.path.join(workspace_path, "args"),
            "logs": os.path.join(workspace_path, "logs"),
            "stats": os.path.join(workspace_path, "stats"),
            "workset": self.workset_path,
            "eval": os.path.join(workspace_path, "evaluation"),
            "analysis": os.path.join(workspace_path, "analysis"),
        }
        [os.makedirs(i, exist_ok=True) for i in structure.values()]
        paths = DotMap(structure)
        return paths

    def load_optimizer(self):
        args_op = self.pr.optimizer.toDict().copy()
        optimizer_type = args_op.pop('name')
        if optimizer_type == 'ADAMW':
            optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, self.model.parameters()), **args_op)
        if self.tr.model == 'HRNET':
            optimizer = get_optimizer(hrnet_config._C, self.model)
        if self.pr.pretrained.use_pretrained:
            model_best_pth = os.path.join(self.paths.checkpoint, 'model_best.pth')
            optimizer.load_state_dict(torch.load(model_best_pth)['optimizer'])
        return optimizer

    def load_model(self):
        if self.tr.model == 'DETR':
            self.logger_cml.report_text(f'Loading DETR Model', level=logging.INFO, print_console=True)
            model = build_model(args=self.pr.detr_args)
            if self.pr.pretrained.use_pretrained:
                model_best_pth = os.path.join(self.paths.checkpoint, 'model_best.pth')
                model_best_state = torch.load(model_best_pth)
                self.logger_cml.report_text(f'Loading model: {model_best_pth}', level=logging.INFO, print_console=True)
                model.load_state_dict(model_best_state['state_dict'], strict=True)
        if self.tr.model == 'HRNET':
            self.logger_cml.report_text(f'Loading HRNET Model', level=logging.INFO, print_console=True)
            config_path = self.pr['model']['HRNET']['config']
            update_config(hrnet_config._C, config_path)
            if self.pr.pretrained.use_pretrained:
                model_best_pth = os.path.join(self.paths.checkpoint, 'model_best.pth')
                model_best_state = torch.load(model_best_pth)
                self.logger_cml.report_text(f'Loading model: {model_best_pth}', level=logging.INFO, print_console=True)
                try:
                    model.load_state_dict(model_best_state['state_dict'].state_dict())
                except:
                    model = model_best_state['state_dict']
            else:
                kwargs = {}
                model = get_face_alignment_net(hrnet_config._C, **kwargs)
        return model.cuda()

    def load_scheduler(self):
        args_sc = self.pr.scheduler.toDict().copy()
        scheduler_type = args_sc.pop('name')
        if scheduler_type == 'StepLR':
            return StepLR(optimizer=self.optimizer, **args_sc)
        elif scheduler_type == 'MultiStepLR':
            return MultiStepLR(optimizer=self.optimizer, **args_sc)

    def backend_operations(self):
        cuda = self.tr.cuda
        torch.version.debug = True if sys.gettrace() else False
        torch.manual_seed(self.tr.torch_seed)
        use_cuda = cuda.use and torch.cuda.is_available()
        device = torch.device(cuda.device_type if use_cuda else "cpu")
        torch.backends.benchmark = self.tr.backend.use_torch
        torch.backends.cudnn.benchmark = True
        torch.set_default_dtype(torch.float32)
        return device

    def train(self):
        # TODO: support multiple gpus
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0]).cuda()
        self.model.to(device=self.device)

        epochs = self.tr['epochs'] + self.last_epoch + 1
        best_nme = 100
        kwargs = dict()
        kwargs.update({'decoder_head': -1})
        kwargs.update({'log_interval': 20})
        kwargs.update({'debug': self.params.single_batch_debug})
        kwargs.update({'model_name': self.tr.model})
        kwargs.update({'decoder_head': -1})
        kwargs.update({'train_with_heatmaps': self.tr.heatmaps.train_with_heatmaps})
        if self.tr.model == 'HRNET':
            kwargs.update({'hm_amp_factor': self.tr['hm_amp_factor']})

        for epoch in range(self.last_epoch + 1, epochs):
            if math.isnan(self.trn_loss) or math.isinf(self.trn_loss):
                break
            if self.train_loader is not None:
                # train
                train_epoch(train_loader=self.train_loader,
                            model=self.model,
                            criteria=self.criteria,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=epoch,
                            logger_cml=self.logger_cml,
                            **kwargs)

                # evaluate
                nme = validate_epoch(val_loader=self.valid_loader,
                                     model=self.model,
                                     criteria=self.criteria,
                                     epoch=epoch,
                                     logger_cml=self.logger_cml,
                                     **kwargs)

            self.scheduler.step()
            self.nnstats.add_measure(epoch, self.model, dump=True)

            is_best = nme < best_nme
            self.logger_cml.report_text(f'is best nme: {is_best}', level=logging.INFO, print_console=True)
            if is_best or epoch % 20 == 0:
                best_nme = min(nme, best_nme)
                self.logger_cml.report_text(f'=> saving checkpoint to {self.paths.checkpoint}',
                                            level=logging.INFO, print_console=True)
                final_model_state_file = os.path.join(self.paths.checkpoint, 'final_state.pth')

                save_checkpoint(states={"state_dict": self.model.state_dict(),
                                        "epoch": epoch + 1,
                                        "best_nme": best_nme,
                                        "optimizer": self.optimizer.state_dict()},
                                is_best=is_best,
                                output_dir=self.paths.checkpoint,
                                filename='checkpoint_{}.pth'.format(epoch))
                self.logger_cml.report_text(f'saving final model state to {final_model_state_file}',
                                            level=logging.INFO, print_console=True)

                torch.save(self.model.state_dict(), final_model_state_file)
