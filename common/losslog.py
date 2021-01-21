import os
import pickle as pk
import time

from torch.utils.tensorboard import SummaryWriter


class CLossLog(object):
    def __init__(self, paths):
        self.logs_path = paths.logs
        self.meta_path = os.path.join(paths.nets, 'meta.pkl')
        self.writer = SummaryWriter(self.logs_path)
        self.meta_model = {}
        self.params_list = []
        self.start_time = time.time()

    def add_value(self, epoch, name, val):
        param_key = name.replace('/', '_')
        timestamp = time.time() - self.start_time
        self.meta_model[epoch] = {param_key: {'time': timestamp, 'val': val}}
        self.writer.add_scalar(param_key, val, epoch, walltime=None)
        self.writer.add_scalar(f'{param_key}_time', val, timestamp, walltime=None)
        self.writer.flush()

    def add_image(self, epoch, name, img):
        self.writer.add_image(name, img, global_step=epoch)
        self.writer.flush()

    def dump(self):
        with open(self.meta_path, "wb") as f:
            pk.dump(self.meta_model, f)

    def load(self, last_epoch=None):
        last_epoch = [] if last_epoch is None else -1
        if last_epoch == -1:
            return

        meta_model = pk.load(open(self.meta_path, "rb"))
        self.start_time = meta_model['starttime']
        self.params_list = meta_model['paramslist']
        self.params = meta_model['params']  # = [epoch, value,timestamp]

        for key in self.params.keys():
            for item in self.params[key]:
                self.writer.add_scalar(self.params_list[key], item[1], item[0], walltime=None)
        self.writer.flush()
        return meta_model

    def get_meta(self):
        meta_model = pk.load(open(self.meta_path, "rb"))
        return meta_model
