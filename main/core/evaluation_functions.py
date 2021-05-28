import logging
import math
import os

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import auc

logger = logging.getLogger(__name__)
# import wandb

from main.core.transforms import transform_preds
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


def get_interocular_distance(pts, num_landmarks=68, box_size=None, tensor=True):
    norm_func = torch.norm if tensor else np.linalg.norm
    if num_landmarks == 19:  # aflw
        interocular = box_size
    elif num_landmarks == 29:  # COFW
        interocular = norm_func(pts[8,] - pts[9,])
    elif num_landmarks == 68:  # 300w
        interocular = norm_func(pts[36,] - pts[45,])
    elif num_landmarks == 98:
        interocular = norm_func(pts[60,] - pts[72,])
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
    ds_err_cpy = ds_err.T[0].copy()
    idx_argmax = ds_err_cpy.argsort()[-num_images_to_analyze:][::-1]
    imgs, preds, tpts = [], [], []

    for b_idx, b_idx_inst in dataset_inst.items():
        [preds.append(b) for b in b_idx_inst['preds']]
        [tpts.append(b.numpy()) for b in b_idx_inst['tpts']]
        [imgs.append(b) for b in b_idx_inst['img']]

    img_plot = [imgs[i] for i in idx_argmax]
    preds_plot = [preds[i] for i in idx_argmax]
    tpts_plot = [tpts[i] for i in idx_argmax]

    analyze_pic = plot_grid_of_ldm(dataset, img_plot, preds_plot, tpts_plot)
    im = Image.fromarray(analyze_pic)
    im.save(os.path.join(output, f'{dataset}_dec_{decoder_head}_analysis_image.png'))


def analyze_results(datastets_inst, datasets, eval_name, output=None, decoder_head=-1):
    logger.info(f"Analyzing results on {eval_name} Datasets")
    datasets = [i.replace("/", "_") for i in datasets]
    tot_err, log = list(), dict()
    for dataset, dataset_inst in datastets_inst.items():
        if dataset not in datasets:
            continue
        logging.info(f"Analysing dataset {dataset} in {eval_name}")
        preds, tpts = list(), list()
        for b_idx, b_idx_inst in dataset_inst.items():
            [preds.append(b.numpy()) for b in b_idx_inst["preds"]]
            [tpts.append(b.numpy()) for b in b_idx_inst["tpts"]]
        nme_ds, auc08_ds, auc10_ds, _ = evaluate_normalized_mean_error(np.array(preds), np.array(tpts))
        log_ds = {dataset: {"auc08": auc08_ds, "auc10": auc10_ds}}
        log.update(log_ds)
        [tot_err.append(i) for i in nme_ds]
        if output is not None:
            save_tough_images(dataset=f'{eval_name.replace(" ", "_")}-{dataset}',
                              dataset_inst=dataset_inst,
                              ds_err=nme_ds,
                              output=output,
                              decoder_head=decoder_head)
    tot_err = np.array(tot_err)
    fail08 = (tot_err > 0.08).mean() * 100
    fail10 = (tot_err > 0.10).mean() * 100
    auc08 = get_auc(tot_err, thresh=0.08) * 100
    auc10 = get_auc(tot_err, thresh=0.10) * 100
    nle = tot_err.mean() * 100

    return {
        "setnick": eval_name,
        "auc08": auc08,
        "auc10": auc10,
        "NLE": nle,
        "fail08": fail08,
        "fail10": fail10,
        "ds_logger": log,
    }


def evaluate_normalized_mean_error(predictions, groundtruth):
    groundtruth = groundtruth.detach().cpu().numpy() if not isinstance(groundtruth, np.ndarray) else groundtruth
    predictions = predictions.detach().cpu().numpy() if not isinstance(predictions, np.ndarray) else predictions

    # compute total average normalized mean error
    assert len(predictions) == len(groundtruth), \
        'The lengths of predictions and ground-truth are not consistent : {} vs {}'.format(len(predictions),
                                                                                           len(groundtruth))
    num_images = len(predictions)

    num_points = predictions.shape[1]
    error_per_image = np.zeros((num_images, 1))
    if num_points == 68:
        interocular_distance_gt = np.linalg.norm(groundtruth[:, 36] - groundtruth[:, 45], axis=1)
    elif num_points == 51 or num_points == 49:
        interocular_distance_gt = np.linalg.norm(groundtruth[:, 19] - groundtruth[:, 28], axis=1)
    else:
        raise Exception('----> Unknown number of points : {}'.format(num_points))
    for i, (pred, gt) in enumerate(zip(predictions, groundtruth)):
        dis_sum = np.linalg.norm(pred - gt, axis=1).sum()
        error_per_image[i] = dis_sum / (num_points * interocular_distance_gt[i])

    # calculate the auc for 0.07/0.08/0.10
    area_under_curve07 = get_auc(error_per_image, 0.07) * 100
    area_under_curve08 = get_auc(error_per_image, 0.08) * 100
    area_under_curve10 = get_auc(error_per_image, 0.10) * 100

    accuracy_under_007 = np.sum(error_per_image < 0.07) * 100. / error_per_image.size
    accuracy_under_008 = np.sum(error_per_image < 0.08) * 100. / error_per_image.size

    for_pck_curve = []
    for x in range(0, 3501, 1):
        error_bar = x * 0.0001
        accuracy = np.sum(error_per_image < error_bar) * 1.0 / error_per_image.size
        for_pck_curve.append((error_bar, accuracy))

    return error_per_image, area_under_curve08, area_under_curve10, for_pck_curve


def get_auc(nme_per_image, thresh=0.08):
    threshold = np.linspace(0, thresh, num=2000)
    accuracys = np.zeros_like(threshold)
    for i in range(threshold.size):
        accuracys[i] = np.sum(nme_per_image.squeeze() < threshold[i]) * 1.0 / nme_per_image.size
    area_under_curve = auc(threshold, accuracys) / thresh
    return area_under_curve
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

