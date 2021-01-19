import globals as g
from train import LDMTrain
from utils.file_handler import FileHandler

PARAMS = 'params.yaml'


class TopLevel(object):
    def __init__(self):
        self.params = self.load_params()
        g.TIMESTAMP = FileHandler.get_datetime()

    def load_params(self):
        return FileHandler.load_yaml(PARAMS)

    def single_epoch_train(self):
        lmd_train = LDMTrain(params=self.params)
        lmd_train.train()
        pass

    def train(self):
        pass

    def find_lr(self):
        pass

    def evaluate(self):
        pass
