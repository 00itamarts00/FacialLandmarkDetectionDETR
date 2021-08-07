from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch


def save_checkpoint(states, output_dir, task_id=None):
    model_best_name = f'{task_id}_model_best.pth' if task_id is not None else 'model_best.pth'
    torch.save(states, os.path.join(output_dir, model_best_name))
