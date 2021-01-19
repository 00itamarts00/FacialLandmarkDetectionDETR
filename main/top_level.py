from train import LDMTrain
from utils.file_handler import FileHandler
PARAMS = 'params.yaml'


class TopLevel(object):
    def __init__(self):
        self.params = FileHandler.load_yaml(PARAMS)

    def single_epoch_train(self):
        lmd_train = LDMTrain(params=self.params)
        lmd_train.train()
        pass

    def train(self):
        pass

    def evaluate(self):
        pass