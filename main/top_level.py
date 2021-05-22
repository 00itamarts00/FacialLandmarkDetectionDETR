import logging
import sys

import main.globals as g
from main.components.evaluator import Evaluator
from main.components.trainer import LDMTrain
from utils.file_handler import FileHandler
from utils.param_utils import *

PARAMS = 'main/params.yaml'


class TopLevel(object):
    def __init__(self, override_params=None):
        self.params = self.load_params()

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
        return fname

    def setup_workspace(self):
        self.setup_pretrained()
        wp = self.params['experiment']['workspace_path']
        name = self.params['experiment']['name']
        ex_workspace_path = os.path.join(wp, name, g.TIMESTAMP)
        os.makedirs(ex_workspace_path, exist_ok=True)
        self.params['workspace_path'] = ex_workspace_path
        FileHandler.save_dict_as_yaml(self.params, os.path.join(ex_workspace_path, 'params.yaml'))

    def setup_pretrained(self, force=False):
        if self.params['experiment']['pretrained']['use_pretrained'] or force:
            g.TIMESTAMP = self.params['experiment']['pretrained']['timestamp']

    def load_params(self):
        return FileHandler.load_yaml(PARAMS)

    def single_batch_train(self):
        self.setup_workspace()
        self.setup_logger(name='single_batch_train')
        override_params = {'train': {'epochs': 150, 'batch_size': 32,
                                     'cuda': {'use': True}
                                     },
                           'experiment': {'single_batch_debug': True}}
        self.params = self.override_params_dict(dict_override=override_params)
        lmd_train = LDMTrain(params=self.params)
        lmd_train.train()

    def train(self):
        self.setup_workspace()
        self.setup_logger(name='train')
        lmd_train = LDMTrain(params=self.params, single_image_train=self.params['experiment']['single_image_train'])
        lmd_train.train()

    def find_learning_rate(self):
        pass

    def evaluate_model(self):
        override_params = {'experiment':
                               {'pretrained':
                                    {'use_pretrained': True,
                                     'timestamp': '220521_145654_16x16_BB2ENC_multidecloss_enc2dec6_refactor'
                                     }
                                }
                           }
        self.params = self.override_params_dict(dict_override=override_params)
        # for dec_head in range(9):
        #     override_params = {'evaluation': {'prediction_from_decoder_head': dec_head}}
        #     self.params = self.override_params_dict(dict_override=override_params)
        #
        #     self.setup_workspace()
        #     self.setup_logger(name='evaluate_model')
        #     lmd_eval = Evaluator(params=self.params)
        #     lmd_eval.evaluate()
        self.setup_workspace()
        self.setup_logger(name='evaluate_model')
        lmd_eval = Evaluator(params=self.params)
        lmd_eval.evaluate()

    def run_experiment(self):
        self.train()
        override_params = {'experiment':
                               {'pretrained':
                                    {'timestamp': g.TIMESTAMP}}}
        self.params = self.override_params_dict(dict_override=override_params)
        self.evaluate_model()
