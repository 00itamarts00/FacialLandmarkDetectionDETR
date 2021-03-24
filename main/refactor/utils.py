from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
from pathlib import Path

import torch
import torch.optim as optim


def save_checkpoint(states, predictions, is_best,
                    output_dir, filename='checkpoint.pth'):
    preds = predictions.cpu().data.numpy()
    torch.save(states, os.path.join(output_dir, filename))
    torch.save(preds, os.path.join(output_dir, 'current_pred.pth'))

    latest_path = os.path.join(output_dir, 'latest.pth')
    if os.path.islink(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.join(output_dir, filename), latest_path)

    if is_best and 'state_dict' in states.keys():
        try:
            torch.save(states.module, os.path.join(output_dir, 'model_best.pth'))
        except:
            torch.save(states, os.path.join(output_dir, 'model_best.pth'))