from __future__ import print_function

import numpy as np
import pandas as pd
import torch

from common.paramutils import get_param


# import wandb
###################################
def calc_accuarcy(epts_batch):
# def calc_accuarcy(dflist):
    mean_err, max_err, std_err = [], [], []
    for key, val in epts_batch.items():
        err = calc_pts_error(np.array(list(val['epts'].values())), np.array(val['opts']))
        mean_err.append(np.mean(err))
        max_err.append(np.max(err))
        std_err.append(np.std(err))
    auc08, fail08, bins, ced68 = calc_CED(mean_err)
    nle = 100 * np.mean(mean_err)
    return auc08, nle, fail08, bins, ced68


def analyze_results(dfresults, datasets, setnick):
    dflist = pd.DataFrame()
    for item in datasets:
        setnick_ = item.replace('/', '_')
        dflist = pd.concat([dflist, dfresults[setnick_]], ignore_index=True)
    auc08, nle, fail08, bins, ced68 = calc_accuarcy(dflist)
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
            epts[i]['epts'] = {k: v/item['sfactor'][i].numpy() for (k, v) in epts[i]['output'].items()}
    return epts


def evaluate_model(device, test_loader, model, workspace_path, config=None):
    if config is None:
        config = {}
    log_interval = get_param(config, 'experiment.log_interval', 20)

    model.eval()

    dflist = pd.DataFrame()
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):
            sample = item['img']
            target = item['hm']
            sample, target = sample.to(device), target.to(device)
            output = model(sample)
            df = model.extract_epts(output, res_factor=1)
            df = add_metadata_to_result(df, item)

            dflist = pd.concat([dflist, df], ignore_index=True)
            if batch_idx % log_interval == 0:
                print(f'Testsing:  [{(batch_idx + 1) * len(sample)}/{len(test_loader.dataset)}'
                      f' ({100. * (batch_idx + 1) / len(test_loader):.02f}%)]')

    return dflist
