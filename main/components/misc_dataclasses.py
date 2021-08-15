import time
from typing import List
from typing import Optional

import numpy as np

from main.core.evaluation_functions import get_auc
from dataclasses import dataclass, field


@dataclass
class BatchEval:
    def __init__(self, epoch, batch_idx):
        self.start_time = time.time()
        self.epoch = epoch
        self.batch_idx = batch_idx
        self.start_time: Optional[float]
        self.end_time: Optional[float]
        self.auc08: Optional[float]
        self.auc10: Optional[float]
        self.loss: Optional[float]
        self.nme: Optional[list]

    @property
    def process_time(self):
        return self.end_time - self.start_time

    def sample_per_sec(self):
        return len(self.nme) / self.process_time

    def avg_losses(self):
        return np.mean(self.loss)

    def end_time(self):
        self.end_time = time.time()


@dataclass
class EpochEval:
    def __init__(self, epoch):
        self.start_time = time.time()
        self.epoch = epoch

    epoch: int
    end_time: float
    batch_eval_lst: List[BatchEval] = field(default_factory=list)

    def end_time(self):
        self.end_time = time.time()

    def average_process_time(self):
        batch_time = self._get_all_values('end_time') - self._get_all_values('start_time')
        return np.mean(batch_time)

    def nme_avg(self):
        return np.mean(self._get_all_values('nme'))

    def loss_avg(self):
        return np.mean(self._get_all_values('loss'))

    def _get_all_values(self, key):
        a = np.array(list(map(lambda x: getattr(x, key), self.batch_eval_lst))).squeeze()
        list_of_list = all(isinstance(elem, np.ndarray) for elem in a)
        if list_of_list:
            res = list()
            [[res.append(i) for i in np.array(ls)] for ls in a]
            return np.array(res) if np.array(res).ndim == 2 else np.array(res).reshape(-1, 1)
        else:
            return a

    def get_failure_rate(self, rate):
        nme_ls = self._get_all_values('nme')
        return (nme_ls > rate).astype(int).mean()

    def get_auc(self, rate):
        nme_ls = self._get_all_values('nme')
        return get_auc(nme_ls, rate) * 100

    def process_time(self):
        return self.end_time - self.start_time

    def get_nle(self):
        return self._get_all_values('nme').mean() * 100

    def __len__(self):
        return len(self._get_all_values('nme'))
