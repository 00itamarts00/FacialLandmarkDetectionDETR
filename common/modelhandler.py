import json
import os
import re
import shutil
from datetime import datetime

import numpy as np
# from pathlib import Path
import torch


class CModelHandler:
    def __init__(self, workspace_path, model, epochs_to_save=None, args=None):
        if args is None:
            args = []
        if epochs_to_save is None:
            epochs_to_save = []
        self.nets_path = os.path.join(workspace_path, 'nets')
        self.args_path = os.path.join(workspace_path, 'args')
        argsbk_path = os.path.join(workspace_path, 'argsbackup')
        self.model = model
        self.epochs_to_save = epochs_to_save
        os.makedirs(self.nets_path, exist_ok=True)
        os.makedirs(self.args_path, exist_ok=True)
        os.makedirs(argsbk_path, exist_ok=True)

        [shutil.move(os.path.join(self.args_path, f), argsbk_path) for f in os.listdir(self.args_path)]

        if len(epochs_to_save) == 0:
            self.epochs_to_save = np.concatenate((range(1, 10, 3), range(10, 100, 10), range(100, 10000, 50)))
        else:
            self.epochs_to_save = epochs_to_save

        now = datetime.now()  # current date and time
        strtime = now.strftime("%m%d%Y_%H%M%S")
        argsfile = os.path.join(self.args_path, f'args_{strtime}.json')
        data = args if isinstance(args, dict) else vars(args)
        with open(argsfile, 'w', encoding='utf-8') as fid:
            json.dump(data, fid, ensure_ascii=False, indent=4)

    def load(self):
        files = [f for f in os.listdir(self.nets_path) if f.endswith('.pt')]
        epidx = [int(re.findall(r'\d+', item)[0]) for item in files]

        if len(epidx) == 0:
            last_epoch = -1
        else:
            last_epoch = max(epidx)
            model_name = ("model_%05d.pt" % (last_epoch))
            self.model.load_state_dict(torch.load(os.path.join(self.nets_path, model_name)))

        return self.model, last_epoch

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def save(self, model, epoch):
        self.model = model
        model_name = ("model_%05d.pt" % (epoch))
        torch.save(model.state_dict(), os.path.join(self.nets_path, model_name))
        last_epoch = max(epoch - 1, 0)
        if (last_epoch not in self.epochs_to_save) and (last_epoch != 0):
            last_model_name = ("model_%05d.pt" % (last_epoch))
            os.remove(os.path.join(self.nets_path, last_model_name))


def load_model(model, nets_path, epoch):
    model_name = ("model_%05d.pt" % (epoch))
    model.load_state_dict(torch.load(os.path.join(nets_path, model_name)))
    return model
