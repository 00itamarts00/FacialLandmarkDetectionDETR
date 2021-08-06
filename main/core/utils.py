from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth', task_id=None):
    torch.save(states, os.path.join(output_dir, filename))

    latest_path = os.path.join(output_dir, 'latest.pth')
    if os.path.islink(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.join(output_dir, filename), latest_path)

    if is_best:
        model_best_name = f'{task_id}_model_best.pth' if task_id is not None else 'model_best.pth'
        torch.save(states, os.path.join(output_dir, model_best_name))
