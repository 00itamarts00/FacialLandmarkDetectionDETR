from __future__ import print_function

import logging
import sys

import numpy as np
import torch

# import wandb
logger = logging.getLogger(__name__)
from main.refactor.evaluation_functions import decode_preds, compute_nme, extract_pts_from_hm


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


def analyze_results(datastets_inst, datasets, setnick):
    logger.info(f'Analyzing results on {setnick} Datasets')
    datasets = [i.replace('/', '_') for i in datasets]
    preds, opts = list(), list()
    mean_err, max_err, std_err = [], [], []
    for dataset, dataset_inst in datastets_inst.items():
        setnick_ = dataset.replace('/', '_')
        if setnick_ not in datasets:
            continue
        for b_idx, b_idx_inst in dataset_inst.items():
            [preds.append(i) for i in b_idx_inst['preds'].numpy()]
            [opts.append(i) for i in b_idx_inst['opts'].numpy()]
    preds = np.squeeze(preds)
    opts = np.squeeze(opts)
    err = compute_nme(preds, opts)
    mean_err.append(np.mean(err))
    max_err.append(np.max(err))
    std_err.append(np.std(err))
    auc08, fail08, bins, ced68 = calc_CED(mean_err)
    nle = 100 * np.mean(mean_err)
    return {'setnick': setnick, 'auc08': auc08, 'NLE': nle, 'fail08': fail08, 'bins': bins, 'ced68': ced68}

    # Test model


def calc_CED(err, x_limit=0.08):
    bins = np.linspace(0, 1, num=10000)
    ced68 = np.zeros(len(bins))
    th_idx = np.argmax(bins >= x_limit)

    for i in range(len(bins)):
        ced68[i] = np.sum(np.array(err) < bins[i]) / len(err)

    auc = 100 * np.trapz(ced68[0:th_idx], bins[0:th_idx]) / x_limit
    failure = 100 * np.sum(np.array(err) > x_limit) / len(err)
    bins_o = bins[0:th_idx]
    ced68_o = ced68[0:th_idx]
    return auc, failure, bins_o, ced68_o


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


def evaluate_model(device, test_loader, model, **kwargs):
    log_interval = kwargs.get('log_interval', 20)
    epts_batch = dict()
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):
            input_, target, opts = item['img'], item['target'], item['opts']
            scale, hm_factor = item['sfactor'], item['hmfactor']
            input_, target = input_.to(device), target.to(device)
            output = model(input_)
            score_map = output.data.cpu()
            preds = extract_pts_from_hm(score_maps=score_map, scale=scale, hm_input_ratio=hm_factor)
            item['preds'] = preds
            epts_batch[batch_idx] = item
            percent = f' ({100. * (batch_idx + 1) / len(test_loader):.02f}%)]'
            sys.stdout.write(f"\rTesting batch {batch_idx}\t{percent}")
            sys.stdout.flush()
    sys.stdout.write(f"\n")
    return epts_batch
