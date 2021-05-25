import os
import pickle as pk
import time
from collections import OrderedDict

from torch.utils.tensorboard import SummaryWriter


class CLossLog(object):
    def __init__(self, paths, last_epoch):
        self.logs_path = paths.logs
        self.last_epoch = last_epoch
        self.meta_path = os.path.join(paths.nets, "meta.pkl")
        self.writer = SummaryWriter(self.logs_path)
        self.meta_model = self.load()
        self.start_time = time.time()

    def add_value(self, epoch, name, val, new_epoch=False):
        if epoch not in self.meta_model.keys():
            self.meta_model[epoch] = dict()
        param_key = name.replace("/", "_")
        timestamp = time.time() - self.start_time
        self.meta_model[epoch].update({param_key: {"time": timestamp, "val": val}})
        self.writer.add_scalar(param_key, val, epoch, walltime=None)
        self.writer.add_scalar(f"{param_key}_time", val, timestamp, walltime=None)
        self.writer.flush()

    def add_image(self, epoch, name, img):
        self.writer.add_image(name, img, global_step=epoch)
        self.writer.flush()

    def dump(self):
        with open(self.meta_path, "wb") as f:
            pk.dump(self.meta_model, f)

    def load(self):
        if self.last_epoch == -1:
            meta_model = OrderedDict()
        else:
            meta_model = self.get_meta()
            for epoch, dc in meta_model.items():
                for param_key, param_val in dc.items():
                    self.writer.add_scalar(
                        epoch, f"{param_key}_time", param_val["time"], walltime=None
                    )
                    self.writer.add_scalar(
                        epoch, f"{param_key}", param_val["val"], walltime=None
                    )
            self.writer.flush()
        return meta_model

    def get_meta(self):
        meta_model = pk.load(open(self.meta_path, "rb"))
        return meta_model
