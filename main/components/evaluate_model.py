from __future__ import print_function

import logging
import sys

import numpy as np
import torch

# import wandb
logger = logging.getLogger(__name__)
from main.refactor.evaluation_functions import calc_CED
from main.refactor.functions import inference


def calc_accuarcy(epts_batch):
    mean_err, max_err, std_err = [], [], []
    for key, val in epts_batch.items():
        err = calc_pts_error(np.array(list(val['epts'].values())), np.array(val['opts']))
        mean_err.append(np.mean(err))
        max_err.append(np.max(err))
        std_err.append(np.std(err))
    auc08, fail08, bins, ced68 = calc_CED(mean_err)
    nle = 100 * np.mean(mean_err)
    return auc08, nle, fail08, bins, ced68


def distance(v1, v2):
    if len(v1.shape) > 1:
        d = np.sqrt(np.sum(np.power(np.subtract(v1, v2), 2), 1))
    else:
        d = np.sqrt(np.sum(np.power(np.subtract(v1, v2), 2)))
    return d


def calc_pts_error(epts, opts, normp=(36, 45)):  # for 68 points
    r = distance(opts[normp[0]], opts[normp[1]])
    d = distance(epts, opts)
    err = np.divide(d, r)
    return err


def add_metadata_to_result(epts, item):
    for key, val in item.items():
        for i, field in enumerate(val):
            epts[i][key] = field
            epts[i]['epts'] = {k: v / item['sfactor'][i].numpy() for (k, v) in epts[i]['output'].items()}
    return epts


def evaluate_model(test_loader, model, decoder_head=-1, **kwargs):
    epts_batch = dict()
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):
            input_, target, opts = item['img'].cuda(), item['target'].cuda(), item['opts'].cuda()
            scale, hm_factor, heatmaps = item['sfactor'].cuda(), item['hmfactor'], item['heatmaps'].cuda()

            output, preds = inference(model, input_batch=input_, scale_factor=scale, **kwargs)

            item['preds'] = [i.cpu().detach() for i in preds]
            epts_batch[batch_idx] = item
            percent = f' ({100. * (batch_idx + 1) / len(test_loader):.02f}%)]'
            sys.stdout.write(f"\rTesting batch {batch_idx}\t{percent}")
            sys.stdout.flush()
    sys.stdout.write(f"\n")
    return epts_batch
