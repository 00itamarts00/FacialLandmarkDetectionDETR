from __future__ import print_function

import logging
import math
import os
import sys

import torch.backends.cudnn
import torch.optim as optim
import wandb
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils import data

import main.globals as g
from main.core.CLMDataset import CLMDataset, get_def_transform, get_data_list
from main.core.functions import train_epoch, validate_epoch
from main.components.nnstats import CnnStats
from main.components.utils import save_checkpoint
from main.detr import detr_args
from main.detr.models.detr import build as build_model
from main.detr.models.detr import load_criteria as load_criteria_detr
from models.HRNET import hrnet_config, update_config
from models.HRNET.HRNET import get_face_alignment_net
from models.HRNET.hrnet_utils import get_optimizer
from utils.file_handler import FileHandler

torch.cuda.empty_cache()
logger = logging.getLogger(__name__)

os.environ["WANDB_API_KEY"] = g.WANDB_API_KEY
# os.environ["WANDB_MODE"] = "dryrun"


# TODO: Load tensorboard logs as df/dict


class LDMTrain(object):
    def __init__(self, params, single_image_train=False):
        self.single_image_train = single_image_train
        self.params = params
        self.workset_path = os.path.join(self.ds['dataset_dir'], self.ds['workset_name'])
        self.paths = self.create_workspace()
        self.last_epoch = self.get_last_epoch()
        self.device = self.backend_operations()
        self.train_loader, self.valid_loader = self.create_dataloaders(datasets=self.tr['datasets']['to_use'])
        self.model = self.load_model()
        self.criteria = self.load_criteria()
        self.optimizer = self.load_optimizer()
        self.scheduler = self.load_scheduler()
        self.nnstats = CnnStats(self.paths.stats, self.model)
        self.writer = self.init_writer()
        self.wandb = self.init_wandb_logger()
        self.trn_loss = 0

    def init_wandb_logger(self):
        # Start a new run, tracking hyperparameters in config
        wandb.init(project="detr_landmark_detection",
                   # group='nothing',
                   config=self.pr
                   )
        # wandb.watch(self.model)
        config = wandb.config
        id = wandb.util.generate_id()
        g.WANDB_INIT = id
        logger.info(f'WandB ID: {id}')
        return wandb

    def get_last_epoch(self):
        meta_path = os.path.join(self.paths.stats, 'meta.pkl')
        if os.path.exists(meta_path):
            meta = FileHandler.load_pkl(meta_path)
            return list(meta.keys())[-1]
        return 0

    def load_criteria(self):
        if self.tr['model'] == 'DETR':
            return load_criteria_detr(args=detr_args)
        if self.tr['model'] == 'HRNET':
            # return Loss_weighted().cuda()
            return torch.nn.MSELoss(size_average=True).cuda()

    @property
    def pr(self):
        return self.params

    @property
    def ds(self):
        return self.pr['dataset']

    @property
    def tr(self):
        return self.pr['train']

    @property
    def ex(self):
        return self.pr['experiment']

    def init_writer(self):
        writer_dict = {
            'writer': SummaryWriter(log_dir=self.paths.logs),
            'train_global_steps': self.last_epoch + 1,
            'valid_global_steps': self.last_epoch + 1,
            'log': {}
        }
        return writer_dict

    def create_dataloaders(self, datasets=None):
        use_cuda = self.tr['cuda']['use']
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

        num_workers = self.tr['cuda']['num_workers'] if sys.gettrace() is None else 0
        kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

        batch_size = self.tr['batch_size'] if not self.ex['single_image_train'] else 1
        train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
        valid_loader = data.DataLoader(validset, batch_size=batch_size, shuffle=False, **kwargs)
        logger.info(f'Number of train images : {len(trainset)}')
        logger.info(f'Number of valid images : {len(validset)}')
        return train_loader, valid_loader

    def create_workspace(self):
        workspace_path = self.pr['workspace_path']
        structure = {'workspace': workspace_path,
                     'checkpoint': os.path.join(workspace_path, 'checkpoint'),
                     'args': os.path.join(workspace_path, 'args'),
                     'logs': os.path.join(workspace_path, 'logs'),
                     'stats': os.path.join(workspace_path, 'stats'),
                     'workset': self.workset_path,
                     'wandb': os.path.join(workspace_path, 'wandb'),
                     'eval': os.path.join(workspace_path, 'evaluation'),
                     'analysis': os.path.join(workspace_path, 'analysis')
                     }
        paths = FileHandler.dict_to_nested_namedtuple(structure)
        [os.makedirs(i, exist_ok=True) for i in paths]
        return paths

    def load_optimizer(self):
        if self.tr['model'] == 'DETR':
            args_op = self.pr['optimizer'][self.tr['optimizer']]
            optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=float(args_op['lr']),
                                    weight_decay=args_op['weight_decay'])
            if self.ex['pretrained']['use_pretrained']:
                model_best_pth = os.path.join(self.paths.checkpoint, 'model_best.pth')
                optimizer.load_state_dict(torch.load(model_best_pth)['optimizer'])
        if self.tr['model'] == 'HRNET':
            optimizer = get_optimizer(hrnet_config._C, self.model)
        return optimizer

    def load_model(self):
        if self.tr['model'] == 'DETR':
            logging.info(f'Loading DETR Model')
            model = build_model(args=detr_args)
            if self.ex['pretrained']['use_pretrained']:
                model_best_pth = os.path.join(self.paths.checkpoint, 'model_best.pth')
                model_best_state = torch.load(model_best_pth)
                logging.info(f'Loading model: {model_best_pth}')
                try:
                    model.load_state_dict(model_best_state['state_dict'].state_dict())
                except:
                    model = model_best_state['state_dict']
        if self.tr['model'] == 'HRNET':
            logging.info(f'Loading HRNET Model')
            config_path = self.pr['model']['HRNET']['config']
            update_config(hrnet_config._C, config_path)
            if self.ex['pretrained']['use_pretrained']:
                model_best_pth = os.path.join(self.paths.checkpoint, 'model_best.pth')
                model_best_state = torch.load(model_best_pth)
                logging.info(f'Loading model: {model_best_pth}')
                try:
                    model.load_state_dict(model_best_state['state_dict'].state_dict())
                except:
                    model = model_best_state['state_dict']
            else:
                kwargs = {}
                model = get_face_alignment_net(hrnet_config._C, **kwargs)
        return model.cuda()

    def load_scheduler(self):
        if self.tr['model'] == 'DETR':
            args_sc = self.pr['scheduler'][self.tr['scheduler']]
            scheduler = StepLR(optimizer=self.optimizer,
                               step_size=1,
                               gamma=0)
        if self.tr['model'] == 'HRNET':
            scheduler = MultiStepLR(optimizer=self.optimizer,
                                    milestones=hrnet_config._C.TRAIN.LR_STEP,
                                    gamma=hrnet_config._C.TRAIN.LR_FACTOR)
        return scheduler

    def backend_operations(self):
        cuda = self.tr['cuda']
        torch.manual_seed(self.tr['torch_seed'])
        use_cuda = cuda['use'] and torch.cuda.is_available()
        device = torch.device(cuda['device_type'] if use_cuda else 'cpu')
        torch.backends.benchmark = self.tr['backend']['use_torch']
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
        kwargs.update({'debug': self.ex['single_batch_debug']})
        kwargs.update({'model_name': self.tr['model']})
        kwargs.update({'decoder_head': -1})
        if self.tr['model'] == 'HRNET':
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
                            epoch=epoch,
                            writer_dict=self.writer,
                            **kwargs)

                # evaluate
                nme = validate_epoch(val_loader=self.valid_loader,
                                     model=self.model,
                                     criteria=self.criteria,
                                     epoch=epoch,
                                     writer_dict=self.writer,
                                     **kwargs)

            self.scheduler.step()
            self.writer['writer'].flush()
            FileHandler.save_dict_to_pkl(self.writer['log'], os.path.join(self.paths.stats, 'meta.pkl'))
            self.nnstats.add_measure(epoch, self.model, dump=True)

            is_best = nme < best_nme
            print(f'is best nme: {is_best}')
            if is_best or epoch % 20 == 0:
                best_nme = min(nme, best_nme)
                logger.info(f'=> saving checkpoint to {self.paths.checkpoint}')
                final_model_state_file = os.path.join(self.paths.checkpoint, 'final_state.pth')

                save_checkpoint(states=
                                {"state_dict": self.model,
                                 "epoch": epoch + 1,
                                 "best_nme": best_nme,
                                 "optimizer": self.optimizer.state_dict()},
                                is_best=is_best,
                                output_dir=self.paths.checkpoint,
                                filename='checkpoint_{}.pth'.format(epoch))

                logger.info(f'saving final model state to {final_model_state_file}')
                torch.save(self.model.state_dict(), final_model_state_file)
            self.writer['writer'].close()
