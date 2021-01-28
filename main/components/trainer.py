from __future__ import print_function

import copy
import math
import os
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CyclicLR

# import shutil
# import json
from packages.detr import detr_args
import time
from packages.detr.models import build_model
from tensorboardX import SummaryWriter
from torch.utils import data
import matplotlib.pyplot as plt
# import pandas as pd
# import torch
# import wandb
from common.losslog import CLossLog
from common.modelhandler import CModelHandler
from main.refactor.nnstats import CnnStats
from main.components.CLMDataset import CLMDataset, get_def_transform, get_data_list
from main.components.evaluate_model import *
from main.components.optimizer import OptimizerCLS
from main.components.scheduler_cls import ScheduleCLS
from main.refactor.functions import train_epoch, validate_epoch
from main.refactor.utils import save_checkpoint
from models import hrnet_config
from models import model_LMDT01, HRNET
from utils.file_handler import FileHandler

torch.cuda.empty_cache()
logger = logging.getLogger(__name__)

# TODO: Load tensorboard logs as df/dict


class LDMTrain(object):
    def __init__(self, params):
        self.pr = params
        self.workset_path = os.path.join(self.ds['dataset_dir'], self.ds['workset_name'])
        self.last_epoch = self.get_last_epoch()
        self.paths = self.create_workspace()
        self.device = self.backend_operations()
        self.train_loader, self.valid_loader = self.create_dataloaders()
        self.model, self.criterion, self.postprocessors = self.load_model()
        self.optimizer = self.load_optimizer()
        self.scheduler = self.load_scheduler()
        self.nnstats = CnnStats(self.paths.stats, self.model)
        self.writer = self.init_writer()
        self.trn_loss = 0
        self.update_last_epoch_in_components()
    
    def get_last_epoch(self):
        return 0

    def update_last_epoch_in_components(self):
        return
    
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
            'train_global_steps': self.last_epoch + 1,
            'valid_global_steps': self.last_epoch + 1,
            'log': {}
        }
        return writer_dict

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

        transform = get_def_transform() if use_augmentations else None

        trainset = CLMDataset(self.pr, self.paths, dftrain, transform=transform)
        validset = CLMDataset(self.pr, self.paths, dfvalid)

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        batch_size = self.tr['batch_size']
        train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
        valid_loader = data.DataLoader(validset, batch_size=batch_size, shuffle=False, **kwargs)

        return train_loader, valid_loader

    def create_workspace(self):
        workspace_path = self.pr['workspace_path']
        structure = {'workspace': workspace_path,
                     'checkpoint': os.path.join(workspace_path, 'checkpoint'),
                     'args': os.path.join(workspace_path, 'args'),
                     'logs': os.path.join(workspace_path, 'logs'),
                     'stats': os.path.join(workspace_path, 'stats'),
                     'workset': self.workset_path
                     }
        paths = FileHandler.dict_to_nested_namedtuple(structure)
        [os.makedirs(i, exist_ok=True) for i in paths]
        return paths

    def load_optimizer(self):
        args_op = self.pr['optimizer'][self.tr['optimizer']]
        optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                                lr=args_op['lr'],
                                weight_decay=args_op['weight_decay'])
        return optimizer

    def load_model(self):
        model, criterion, postprocessors = build_model(args=detr_args)
        return model, criterion, postprocessors

    def load_scheduler(self):
        args_sc = self.pr['scheduler'][self.tr['scheduler']]
        scheduler = StepLR(optimizer=self.optimizer,
                           step_size=args_sc['step_size'],
                           gamma=args_sc['gamma'])
        return scheduler

    def backend_operations(self):
        cuda = self.tr['cuda']
        torch.manual_seed(self.tr['torch_seed'])
        use_cuda = cuda['use'] and torch.cuda.is_available()
        device = torch.device(cuda['device_type'] if use_cuda else 'cpu')
        torch.backends.benchmark = self.tr['backend']['use_torch']
        return device

    def train(self):
        run_valid = self.tr['run_valid']

        # TODO: support multiple gpus
        self.model = torch.nn.DataParallel(self.model, device_ids=[0]).cuda()

        epochs = self.tr['epochs'] + self.last_epoch + 1
        best_nme = 100
        nme = 0
        predictions = None
        for epoch in range(self.last_epoch + 1, epochs):
            if math.isnan(self.trn_loss) or math.isinf(self.trn_loss):
                break
            if self.train_loader is not None:
                starttime = time.time()
                # train
                kwargs = {'log_interval': 20,
                          'debug': self.ex['single_batch_debug']}
                train_epoch(train_loader=self.train_loader,
                            model=self.model,
                            criterion=self.criterion,
                            optimizer=self.optimizer,
                            epoch=epoch,
                            writer_dict=self.writer,
                            **kwargs)

                # evaluate
                kwargs = {'num_landmarks': self.tr['num_landmarks'],
                          'debug': self.ex['single_batch_debug']}
                nme, predictions = validate_epoch(val_loader=self.valid_loader,
                                                  model=self.model,
                                                  criterion=self.criterion,
                                                  epoch=epoch,
                                                  writer_dict=self.writer,
                                                  **kwargs)

            self.scheduler.step()
            self.writer['writer'].flush()
            FileHandler.save_dict_to_pkl(self.writer['log'], os.path.join(self.paths.stats, 'meta.pkl'))
            self.nnstats.add_measure(epoch, self.model, dump=True)

            is_best = nme < best_nme
            print(f'is best nme: {is_best}')
            best_nme = min(nme, best_nme)
            if is_best or epoch % 20 == 0:
                logger.info(f'=> saving checkpoint to {self.paths.checkpoint}')
                final_model_state_file = os.path.join(self.paths.checkpoint, 'final_state.pth')

                save_checkpoint(states=
                                {"state_dict": self.model,
                                 "epoch": epoch + 1,
                                 "best_nme": best_nme,
                                 "optimizer": self.optimizer.state_dict()},
                                predictions=predictions,
                                is_best=is_best,
                                output_dir=self.paths.checkpoint,
                                filename='checkpoint_{}.pth'.format(epoch))

                logger.info(f'saving final model state to {final_model_state_file}')
                torch.save(self.model.state_dict(), final_model_state_file)
            self.writer['writer'].close()



