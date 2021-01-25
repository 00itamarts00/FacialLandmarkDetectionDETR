import json
import os
import re
import numpy as np
import torch


class CModelHandler(object):
    def __init__(self, model, checkpoint_path, args, **kwargs):
        self.checkpoint = checkpoint_path
        self.args_path = args
        self.last_epoch = self.get_last_epoch()
        self.model = model
        if self.last_epoch != -1:
            self.load_pretrained()
        self.epochs_to_save = np.concatenate((range(1, 10, 3), range(10, 100, 10), range(100, 10000, 50)))\
            if kwargs.get('epochs_to_save') is None else kwargs.get('epochs_to_save')

    def get_last_epoch(self):
        files = [f for f in os.listdir(self.checkpoint) if f.endswith('.pt')]
        epidx = [int(re.findall(r'\d+', item)[0]) for item in files]
        last_epoch = -1 if len(epidx) == 0 else max(epidx)
        return last_epoch

    def load_pretrained(self):
        model_name = f'model_{str(self.last_epoch).zfill(5)}.pt'
        pretrained_dict = torch.load(os.path.join(self.checkpoint, model_name))
        self.model.load_state_dict(pretrained_dict)

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
        torch.save(model.state_dict(), os.path.join(self.checkpoint, model_name))
        last_epoch = max(epoch - 1, 0)
        if (last_epoch not in self.epochs_to_save) and (last_epoch != 0):
            last_model_name = f'model_{str(last_epoch).zfill(5)}.pt'
            os.remove(os.path.join(self.checkpoint, last_model_name))


def load_model(model, checkpoint, epoch):
    model_name = f'model_{str(epoch).zfill(5)}.pt'
    model.load_state_dict(torch.load(os.path.join(checkpoint, model_name)))
    return model
