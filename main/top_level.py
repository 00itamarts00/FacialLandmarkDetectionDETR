import main.globals as g
from main.components.trainer import LDMTrain
from utils.file_handler import FileHandler
from utils.param_utils import *
import logging
import sys
import cProfile
import inspect
PARAMS = 'main/params.yaml'


class TopLevel(object):
    def __init__(self, override_params=None):
        self.params = self.load_params()
        self.setup_pretrained()
        self.setup_workspace()

    def override_params_dict(self, dict_override):
        if dict_override is None or dict_override == dict():
            return self.params
        else:
            return update_nested_dict(self.params, dict_override)

    def setup_logger(self, name):
        fname = os.path.join(self.params['workspace_path'], name)
        # noinspection PyArgumentList
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            handlers=[logging.FileHandler(fname + '.log'),
                                      logging.StreamHandler(sys.stdout)])
        logger = logging.getLogger(__name__)
        logger.info('Initiate Logger')

    def setup_workspace(self):
        wp = self.params['experiment']['workspace_path']
        name = self.params['experiment']['name']
        ex_workspace_path = os.path.join(wp, name, g.TIMESTAMP)
        os.makedirs(ex_workspace_path, exist_ok=True)
        self.params['workspace_path'] = ex_workspace_path
        FileHandler.save_dict_as_yaml(self.params, os.path.join(ex_workspace_path, 'params.yaml'))

    def setup_pretrained(self):
        if self.params['experiment']['pretrained']['use_pretrained']:
            g.TIMESTAMP = self.params['experiment']['pretrained']['timestamp']

    def load_params(self):
        return FileHandler.load_yaml(PARAMS)

    def single_batch_train(self):
        self.setup_logger(name='single_batch_train')
        override_params = {'train': {'epochs': 5, 'batch_size': 20},
                           'experiment': {'single_batch_debug': True}}
        self.params = self.override_params_dict(dict_override=override_params)
        lmd_train = LDMTrain(params=self.params)
        lmd_train.train()

    def single_epoch_train(self):
        self.setup_logger(name='single_epoch_train')
        override_params = {'train': {'epochs': 1}}
        self.params = self.override_params_dict(dict_override=override_params)
        lmd_train = LDMTrain(params=self.params)
        lmd_train.train()

    def train(self):
        self.setup_logger(name='train')
        lmd_train = LDMTrain(params=self.params)
        lmd_train.train()
        pass

    def find_learning_rate(self):
        pass

    def evaluate(self):
        pass
