import os
import sys

import clearml
from clearml import Task
from dotmap import DotMap
from pygit2 import Repository
import main.globals as g
from main.components.evaluator import Evaluator
from main.components.trainer import LDMTrain
from common.s3_interface import upload_file_to_s3
from models.TRANSPOSE.utils import get_transpose_params
from utils.file_handler import FileHandler
from utils.param_utils import *
from clearml.logger import Logger

PARAMS = 'main/params.yaml'
DETR_ARGS = 'main/detr/detr_args.yaml'
PERC_ARGS = 'models/PERCIEVER/perciever_args.yaml'
TRANSPOSE_ARGS = 'models/TRANSPOSE/transpose_args.yaml'
SCHEDULER_PARAMS = 'main/scheduler_params.yaml'
OPTIMIZER_PARAMS = 'main/optimizer_params.yaml'


class TopLevel(object):
    def init(self, task_id=None):
        self.params = self.load_params()
        if self.params.pretrained.use_pretrained or task_id is not None:
            task_id = task_id if task_id is not None else self.params.pretrained.task_id
            self.task = Task.get_task(task_id=task_id)
            g.TASK_ID = self.task.task_id
            self.params = self.load_params(self.task.task_id)
            self.params.pretrained.use_pretrained = True
            self.logger = self.task.logger
        else:
            self.task = self.init_clearml()

    def init_clearml(self):
        project_name = self.params.project if not sys.gettrace() else 'DEBUG'
        # TODO: change task name to custom value
        task_name = self.get_current_git_branch_name()
        task = Task.init(project_name, task_name, task_type='training', reuse_last_task_id=False)
        self.params.update(task_id=task.task_id)
        g.TASK_ID = task.task_id
        task.connect(self.params)
        tags = [str(g.TIMESTAMP)] if not sys.gettrace() else [str(g.TIMESTAMP), 'DEBUG']
        task.set_tags(tags)
        self.logger = Logger.current_logger()
        return task

    @staticmethod
    def get_current_git_branch_name():
        return Repository('.').head.shorthand

    def override_params_dict(self, dict_override):
        if dict_override is None or dict_override == dict():
            return self.params
        else:
            return DotMap(update_nested_dict(self.params.toDict(), dict_override))

    def setup_workspace(self):
        self.setup_pretrained()
        wp = self.params.workspace_path
        name = self.params.train.model
        ex_workspace_path = os.path.join(wp, name, str(g.TIMESTAMP))
        os.makedirs(ex_workspace_path, exist_ok=True)
        self.params.workspace_path = ex_workspace_path
        FileHandler.save_dict_as_yaml(self.params.toDict(), os.path.join(ex_workspace_path, 'params.yaml'))

    def setup_pretrained(self, force=False):
        if self.params.pretrained.use_pretrained or force:
            g.TIMESTAMP = self.task.get_tags()[0]

    def load_params(self, task_id=None):
        if task_id is None:
            params = DotMap(FileHandler.load_yaml(PARAMS))
            params = self.integrate_optimizer_params(params)
            params = self.integrate_scheduler_params(params)
            params = self.integrate_model_params(params)
        else:
            params = DotMap(self.task.get_parameters_as_dict()['General'])
            params.pretrained.use_pretrained = True
            params.pretrained.task_id = task_id
            # this is to fix a bug with the dictionary parsing
            params = DotMap(fix_parsing_values_to_int(params.toDict()))
        return params

    @staticmethod
    def integrate_optimizer_params(params):
        optimizer_params = FileHandler.load_yaml(OPTIMIZER_PARAMS)['optimizer']
        dc_op = {'optimizer': {'name': params.train.optimizer}}
        dc_op['optimizer'].update(optimizer_params[params.train.optimizer])
        return DotMap(update_nested_dict(params.toDict(), dc_op))

    @staticmethod
    def integrate_scheduler_params(params):
        scheduler_params = FileHandler.load_yaml(SCHEDULER_PARAMS)['scheduler']
        dc_sc = {'scheduler': {'name': params.train.scheduler}}
        dc_sc['scheduler'].update(scheduler_params[params.train.scheduler])
        return DotMap(update_nested_dict(params.toDict(), dc_sc))

    @staticmethod
    def integrate_model_params(params):
        if params.train.model == 'DETR':
            return DotMap(update_nested_dict(params.toDict(), FileHandler.load_yaml(DETR_ARGS)))
        elif params.train.model == 'HRNET':
            raise NotImplementedError
        # TODO: update params with HRNET config params
        elif params.train.model == 'PERC':
            return DotMap(update_nested_dict(params.toDict(), FileHandler.load_yaml(PERC_ARGS)))
        elif params.train.model == 'TRANSPOSE':
            transpose_params = get_transpose_params(FileHandler.load_yaml(TRANSPOSE_ARGS))
            return DotMap(update_nested_dict(params.toDict(), transpose_params))

    def single_batch_train(self):
        self.init()
        self.setup_workspace()
        override_params = {'train': {'epochs': 150, 'batch_size': 32,
                                     'cuda': {'use': True}
                                     },
                           'experiment': {'single_batch_debug': True}}
        self.params = self.override_params_dict(dict_override=override_params)
        lmd_train = LDMTrain(params=self.params, last_epoch=self.task.get_last_iteration(), logger=self.logger)
        lmd_train.train()

    def train(self, task_id):
        self.init(task_id=task_id)
        self.setup_workspace()
        lmd_train = LDMTrain(params=self.params, last_epoch=self.task.get_last_iteration(), logger=self.logger,
                             task_id=self.task.task_id)
        lmd_train.train()
        upload_file_to_s3(file_name=lmd_train.model_best_pth)

    def evaluate_model(self, task_id):
        self.init(task_id=task_id)
        self.setup_workspace()
        lmd_eval = Evaluator(params=self.params, logger_cml=self.task.logger,
                             task_id=self.task.task_id)
        lmd_eval.evaluate()

    def run_experiment(self):
        self.train()
        override_params = {'experiment':
                               {'pretrained':
                                    {'timestamp': str(g.TIMESTAMP)}}}
        self.params = self.override_params_dict(dict_override=override_params)
        self.evaluate_model()
