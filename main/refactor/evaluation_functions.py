# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import logging
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)
# import wandb

from main.refactor.transforms import transform_preds
from utils.plot_utils import plot_grid_of_ldm


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


def compute_nme(preds, opts, box_size=None, tensor=True):
    norm_func = torch.norm if tensor else np.linalg.norm
    sum_func = torch.sum if tensor else np.sum
    batch_size = preds.shape[0]
    num_landmarks = preds.shape[1]
    interoculars = [get_interocular_distance(pts_gt, num_landmarks=num_landmarks, tensor=tensor) for pts_gt in opts]
    rmse_vec = [norm_func(preds[i, ] - opts[i, ]) for i in range(batch_size)]
    if tensor:
        nme = sum_func(torch.Tensor(rmse_vec).cuda()) / (torch.Tensor(interoculars).cuda() * num_landmarks)
    else:
        nme = sum_func(np.array(rmse_vec)) / np.array(interoculars * num_landmarks)
    return nme


def get_interocular_distance(pts, num_landmarks=68, box_size=None, tensor=True):
    norm_func = torch.norm if tensor else np.linalg.norm
    if num_landmarks == 19:   # aflw
        interocular = box_size
    elif num_landmarks == 29:  # COFW
        interocular = norm_func(pts[8, ] - pts[9, ])
    elif num_landmarks == 68:  # 300w
        interocular = norm_func(pts[36, ] - pts[45, ])
    elif num_landmarks == 98:
        interocular = norm_func(pts[60, ] - pts[72, ])
    else:
        raise ValueError('Number of landmarks is wrong')
    return interocular


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


def save_tough_images(dataset, dataset_inst, ds_err, output, decoder_head=-1):
    num_images_to_analyze = 12
    idx_argmax = ds_err.argsort()[-num_images_to_analyze:][::-1]
    imgs, sfactor, preds, opts = [], [], [], []

    for b_idx, b_idx_inst in dataset_inst.items():
        [preds.append(b) for b in b_idx_inst['preds']]
        [opts.append(b.numpy()) for b in b_idx_inst['opts']]
        [sfactor.append(b.numpy()) for b in b_idx_inst['sfactor']]
        [imgs.append(b) for b in b_idx_inst['img']]

    img_plot = [imgs[i] for i in idx_argmax]
    preds_plot = [preds[i] for i in idx_argmax]
    opts_plot = [opts[i] for i in idx_argmax]
    sfactor_plot = [sfactor[i] for i in idx_argmax]

    analyze_pic = plot_grid_of_ldm(dataset, img_plot, preds_plot, opts_plot, sfactor_plot)
    im = Image.fromarray(analyze_pic)
    im.save(os.path.join(output, f'{dataset}_dec_{decoder_head}_analysis_image.png'))


def analyze_results(datastets_inst, datasets, setnick, output=None, decoder_head=-1):
    logger.info(f'Analyzing results on {setnick} Datasets')
    datasets = [i.replace('/', '_') for i in datasets]
    mean_err, max_err, std_err = [], [], []
    tot_err = list()
    for dataset, dataset_inst in datastets_inst.items():
        preds, opts = list(), list()
        setnick_ = dataset.replace('/', '_')
        if setnick_ not in datasets:
            continue
        for b_idx, b_idx_inst in dataset_inst.items():
            [preds.append(b.numpy()) for b in b_idx_inst['preds']]
            [opts.append(b.numpy()) for b in b_idx_inst['opts']]
        preds = np.array([np.array(i) for i in preds])
        ds_err = compute_nme(np.array(preds), np.array(opts))
        [tot_err.append(i) for i in ds_err]
        mean_err.append(np.mean(ds_err))
        max_err.append(np.max(ds_err))
        std_err.append(np.std(ds_err))
        if output is not None:
            save_tough_images(f'{setnick.replace(" ", "_")}-{dataset}', dataset_inst, ds_err, output,
                              decoder_head=decoder_head)
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


def evaluate_model(device, test_loader, model, decoder_head=-1, **kwargs):
    log_interval = kwargs.get('log_interval', 20)
    epts_batch = dict()
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):
            input_, target, opts = item['img'], item['target'], item['opts']
            scale, hm_factor = item['sfactor'], item['hmfactor']
            input_, target = input_.to(device), target.to(device)
            output = model(input_)
            output_coords = output['pred_coords'].cpu().detach().numpy()[decoder_head] if output[
                                                                                              'pred_coords'].dim() == 4 else \
                output['pred_coords'].cpu().detach().numpy()
            preds = output_coords * 256
            item['preds'] = preds
            item['preds'] = [i / s for (i, s) in zip(preds, scale)]
            # item['opts'] = [i * s for (i, s) in zip(opts, scale)]
            epts_batch[batch_idx] = item
            percent = f' ({100. * (batch_idx + 1) / len(test_loader):.02f}%)]'
            sys.stdout.write(f"\rTesting batch {batch_idx}\t{percent}")
            sys.stdout.flush()
    sys.stdout.write(f"\n")
    return epts_batch
