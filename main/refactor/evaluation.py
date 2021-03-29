# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import logging
import math
import sys
from tqdm import tqdm
import numpy as np
import torch

logger = logging.getLogger(__name__)
# import wandb

from main.refactor.transforms import transform_preds


def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def compute_nme(preds, opts, box_size=None):
    target = opts

    batch_size = preds.shape[0]
    num_landmarks = preds.shape[1]
    rmse = np.zeros(batch_size)

    for i in range(batch_size):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if num_landmarks == 19:  # aflw
            interocular = box_size
        elif num_landmarks == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8,] - pts_gt[9,])
        elif num_landmarks == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36,] - pts_gt[45,])
        elif num_landmarks == 98:
            interocular = np.linalg.norm(pts_gt[60,] - pts_gt[72,])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * num_landmarks)

    return rmse


def extract_pts_from_hm(score_maps, scale, hm_input_ratio):
    pred = [None] * score_maps.shape[0]
    for k, hm_stack in enumerate(score_maps):
        pts = [None] * score_maps.shape[1]
        for p, hm in enumerate(hm_stack):
            max_idx = np.unravel_index(np.argmax(hm, axis=None), hm.shape)  # returns a tuple
            pttmp = np.array(max_idx)
            pts[p] = np.array([pttmp[1], pttmp[0]])
        pts = np.multiply(np.multiply(pts, 1 / scale.numpy()[k]), hm_input_ratio.numpy()[k])
        pred[k] = pts
    return torch.tensor(pred)


def decode_preds(output, center, scale, res):
    coords = get_preds(output)  # float type

    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1] - hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds


def analyze_results(datastets_inst, datasets, setnick):
    logger.info(f'Analyzing results on {setnick} Datasets')
    datasets = [i.replace('/', '_') for i in datasets]
    preds, opts = list(), list()
    mean_err, max_err, std_err = [], [], []
    tot_err = list()
    for dataset, dataset_inst in datastets_inst.items():
        setnick_ = dataset.replace('/', '_')
        if setnick_ not in datasets:
            continue
        for b_idx, b_idx_inst in dataset_inst.items():
            [preds.append(b) for b in b_idx_inst['preds']]
            [opts.append(b.numpy()) for b in b_idx_inst['opts']]
        err = compute_nme(np.array(preds), np.array(opts))
        [tot_err.insert(0, i) for i in err]
        mean_err.append(np.mean(err))
        max_err.append(np.max(err))
        std_err.append(np.std(err))
    tot_err = np.array(tot_err)
    auc08, fail08, bins, ced68 = calc_CED(tot_err)
    nle = 100 * np.mean(tot_err)
    return {'setnick': setnick, 'auc08': auc08, 'NLE': nle, 'fail08': fail08, 'bins': bins, 'ced68': ced68}


def calc_CED(err, x_limit=0.08):
    bins = np.linspace(0, 1, num=10000)
    ced68 = np.zeros(len(bins))
    th_idx = np.where(bins >= x_limit)[0][0]

    for i in range(len(bins)):
        ced68[i] = np.sum(np.array(err) < bins[i]) / len(err)

    auc = 100 * np.trapz(ced68[0:th_idx], bins[0:th_idx]) / x_limit
    failure = 100 * np.sum(np.array(err) > x_limit) / len(err)
    bins_o = bins[0:th_idx]
    ced68_o = ced68[0:th_idx]
    return auc, failure, bins_o, ced68_o


def evaluate_model(device, test_loader, model, **kwargs):
    log_interval = kwargs.get('log_interval', 20)
    epts_batch = dict()
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):
            input_, target, opts = item['img'], item['target'], item['opts']
            scale, hm_factor = item['sfactor'], item['hmfactor']
            input_, target = input_.to(device), target.to(device)
            output = model(input_)
            preds = output['pred_coords'].cpu().detach().numpy() * 256
            item['preds'] = preds
            item['opts'] = [i * s for (i, s) in zip(opts, scale)]
            epts_batch[batch_idx] = item
            percent = f' ({100. * (batch_idx + 1) / len(test_loader):.02f}%)]'
            sys.stdout.write(f"\rTesting batch {batch_idx}\t{percent}")
            sys.stdout.flush()
    sys.stdout.write(f"\n")
    return epts_batch
