import os
import pickle as pk
import time

from torch.utils.tensorboard import SummaryWriter


class CLossLog(object):
    def __init__(self, paths):
        self.logs_path = paths.logs
        self.meta_path = os.path.join(paths.nets, 'meta.pkl')
        self.writer = SummaryWriter(self.logs_path)
        self.params = {}
        self.paramslist = {}
        self.starttime = time.time()

    def add_value(self, epoch, name, val):
        param_key = name.replace('/', '_')
        if not (param_key in self.paramslist):
            self.paramslist[param_key] = name
            self.params[param_key] = []

        tstamp = time.time() - self.starttime
        self.params[param_key].append([epoch, val, tstamp])
        self.writer.add_scalar(self.paramslist[param_key], val, epoch, walltime=None)
        self.writer.add_scalar(f'{self.paramslist[param_key]}_time', val, tstamp, walltime=None)
        self.writer.flush()

    def add_image(self, epoch, name, img):
        self.writer.add_image(name, img, global_step=epoch)
        self.writer.flush()

    def dump(self):
        meta_model = {'paramslist': self.paramslist, 'params': self.params, 'starttime': self.starttime}
        with open(self.meta_path, "wb") as f:
            pk.dump(meta_model, f)

    def load(self, last_epoch=None):
        if last_epoch is None:
            last_epoch = []
        if last_epoch == -1:
            return

        meta_model = pk.load(open(self.meta_path, "rb"))
        self.starttime = meta_model['starttime']
        self.paramslist = meta_model['paramslist']
        self.params = meta_model['params']  # = [epoch, value,timestamp]

        for key in self.params.keys():
            for item in self.params[key]:
                self.writer.add_scalar(self.paramslist[key], item[1], item[0], walltime=None)

        self.writer.flush()

        return meta_model

    def get_meta(self):
        meta_model = pk.load(open(self.meta_path, "rb"))
        return meta_model
