import main.globals as g
from main.components.trainer import LDMTrain
from utils.file_handler import FileHandler
from utils.param_utils import *

PARAMS = 'main/params.yaml'


class TopLevel(object):
    def __init__(self, override_params=None):
        self.params = self.load_params()
        self.setup_pretrained()

    def override_params_dict(self, dict_override):
        if dict_override is None or dict_override == dict():
            return self.params
        else:
            return update_nested_dict(self.params, dict_override)

    def setup_pretrained(self):
        if self.params['experiment']['pretrained']['use_pretrained']:
            g.TIMESTAMP = self.params['experiment']['pretrained']['timestamp']

    def load_params(self):
        return FileHandler.load_yaml(PARAMS)

    def single_image_train(self):
        override_params = {'train': {'epochs': 50}}
        self.params = self.override_params_dict(dict_override=override_params)
        lmd_train = LDMTrain(params=self.params)
        lmd_train.train()

    def single_epoch_train(self):
        override_params = {'train': {'epochs': 1}}
        self.params = self.override_params_dict(dict_override=override_params)
        lmd_train = LDMTrain(params=self.params)
        lmd_train.train()

    def train(self):
        lmd_train = LDMTrain(params=self.params)
        lmd_train.train()
        pass

    def find_learning_rate(self):
        pass

    def evaluate(self):
        pass
