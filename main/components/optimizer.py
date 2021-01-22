from packages.Ranger_Deep_Learning_Optimizer_master.ranger import Ranger  # this is from ranger.py
import torch.optim as optim
import numpy as np


class OptimizerCLS(object):
    def __init__(self, params, model):
        self.params = params
        self.model = model
        self.optimizer_type = self.params['train']['optimizer']
        self.trainable_parameters = self.get_trainable_parameters()

    @property
    def op(self): return self.params['optimizer'][self.optimizer_type]

    def load_optimizer(self):
        if self.optimizer_type == 'RANGER':
            return self.load_ranger_opt()
        if self.optimizer_type == 'ADAM':
            return self.load_adam_opt()
        if self.optimizer_type == 'SGD':
            return self.load_sgd_opt()

    def get_trainable_parameters(self):
        # TODO: understand this better - itamar
        trainable_parameters = self.params['train']['trainable_parameters']
        if trainable_parameters is None:
            return self.model.parameters()
        else:       # TODO: this addition is for training small additions to the net
            params = []
            for name, p in self.model.named_parameters():
                if trainable_parameters in name:
                    print(f'{name} is going to be trained')
                    params.append(p)
                else:
                    print(f'{name} is going to be fixed')
            if len(params) == 0:
                print(f'Warning !!!! - Paramteres: {trainable_parameters} - Are not included in model')
                params = self.model.parameters()
                return params

    def load_ranger_opt(self):
        optimizer = Ranger(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                           lr=self.op['lr'],
                           alpha=self.op['alpha'],
                           K=self.op['k'],
                           N_sma_threshhold=self.op['N_sma_threshhold'],
                           betas=self.op['betas'],
                           eps=self.op['eps'],
                           weight_decay=self.op['weight_decay'],
                           use_gc=self.op['use_gc'],
                           gc_conv_only=self.op['gc_conv_only'])
        return optimizer

    def load_sgd_opt(self):
        optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                              lr=self.op['lr'],
                              momentum=self.op['momentum'],
                              dampening=self.op['dampening'],
                              weight_decay=self.op['weight_decay'],
                              nesterov=self.op['nesterov'])
        return optimizer

    def load_adam_opt(self):
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=self.op['lr'],
                               betas=self.op['betas'],
                               eps=float(self.op['eps']),
                               weight_decay=self.op['weight_decay'],
                               amsgrad=self.op['amsgrad'])
        return optimizer

