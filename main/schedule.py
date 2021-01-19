from torch.optim.lr_scheduler import StepLR, CyclicLR


class ScheduleCLS(object):
    def __init__(self, params, optimizer):
        self.params = params
        self.optimizer = optimizer
        self.scheduler_type = self.params['train']['scheduler']

    @property
    def sc(self): return self.params['scheduler'][self.scheduler_type]

    def load_scheduler(self):
        if self.scheduler_type == 'StepLR':
            return self.load_steplr_scheduler()
        if self.scheduler_type == 'CyclicLR':
            return self.load_cyclic_lr_scheduler()

    def load_steplr_scheduler(self):
        scheduler = StepLR(optimizer=self.optimizer,
                           step_size=self.sc['step_size'],
                           gamma=self.sc['gamma'])
        return scheduler

    def load_cyclic_lr_scheduler(self):
        scheduler = CyclicLR(optimizer=self.optimizer,
                             base_lr=self.sc['base_lr'],
                             max_lr=self.sc['max_lr'],
                             step_size_up=self.sc['step_size_up'])
        return scheduler

# step_size = get_param(config, 'train.step_size', 200)
    # gamma = get_param(config, 'train.gamma', 0.3)
    # scheduler_type = get_param(config, 'train.scheduler_type', 'StepLR')
    #
    # if scheduler_type == 'StepLR':
    #     scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    # elif scheduler_type == 'CyclicLR':
    #     scheduler = CyclicLR(optimizer, base_lr=get_param(config, 'train.CyclicLR.base_lr', 1e-3),
    #                          max_lr=get_param(config, 'train.CyclicLR.max_lr', 1e-1),
    #                          step_size_up=get_param(config, 'train.CyclicLR.step_size_up', 100))
    #
    # for idx in range(0, last_epoch):
    #     scheduler.step()
    #
