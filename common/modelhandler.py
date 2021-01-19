import json
import os
import re
import shutil
from datetime import datetime
from utils.file_handler import FileHandler
import numpy as np
import torch


class CModelHandler(object):
    def __init__(self, model, **kwargs):
        self.model = model
        self.nets_path = os.path.join(kwargs.get('workspace_path'), 'nets')
        self.args_path = os.path.join(kwargs.get('workspace_path'), 'args')
        self.epochs_to_save = np.concatenate((range(1, 10, 3), range(10, 100, 10), range(100, 10000, 50)))\
            if kwargs.get('epochs_to_save') is None else kwargs.get('epochs_to_save')
        self._create_folders()

    def _create_folders(self):
        os.makedirs(self.nets_path, exist_ok=True)
        os.makedirs(self.args_path, exist_ok=True)

    def load(self):
        files = [f for f in os.listdir(self.nets_path) if f.endswith('.pt')]
        epidx = [int(re.findall(r'\d+', item)[0]) for item in files]
        last_epoch = -1 if len(epidx) == 0 else max(epidx)
        model_name = f'model_{str(last_epoch).zfill(5)}.pt'
        model = self.model if len(epidx) == 0 else\
            self.model.load_state_dict(torch.load(os.path.join(self.nets_path, model_name)))
        return model, last_epoch

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        return m

    def save(self, model, epoch):
        self.model = model
        model_name = f'model_{str(epoch).zfill(5)}.pt'
        torch.save(model.state_dict(), os.path.join(self.nets_path, model_name))
        last_epoch = max(epoch - 1, 0)
        if (last_epoch not in self.epochs_to_save) and (last_epoch != 0):
            last_model_name = f'model_{str(last_epoch).zfill(5)}.pt'
            os.remove(os.path.join(self.nets_path, last_model_name))


def load_model(model, nets_path, epoch):
    model_name = f'model_{str(epoch).zfill(5)}.pt'
    model.load_state_dict(torch.load(os.path.join(nets_path, model_name)))
    return model
