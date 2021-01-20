from torch.optim.lr_scheduler import StepLR, CyclicLR


class ScheduleCLS(object):
    def __init__(self, params, optimizer, last_epoch):
        self.params = params
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.scheduler_type = self.params['train']['scheduler']

    @property
    def sc(self): return self.params['scheduler'][self.scheduler_type]

    def advance_scheduler_to_last_epoch(self, scheduler):
        for idx in range(0, self.last_epoch):
            scheduler.step()

    def load_scheduler(self):
        if self.scheduler_type == 'StepLR':
            return self.load_steplr_scheduler()
        if self.scheduler_type == 'CyclicLR':
            return self.load_cyclic_lr_scheduler()

    def load_steplr_scheduler(self):
        scheduler = StepLR(optimizer=self.optimizer,
                           step_size=self.sc['step_size'],
                           gamma=self.sc['gamma'])
        self.advance_scheduler_to_last_epoch(scheduler)
        return scheduler

    def load_cyclic_lr_scheduler(self):
        scheduler = CyclicLR(optimizer=self.optimizer,
                             base_lr=self.sc['base_lr'],
                             max_lr=self.sc['max_lr'],
                             step_size_up=self.sc['step_size_up'])
        self.advance_scheduler_to_last_epoch(scheduler)
        return scheduler