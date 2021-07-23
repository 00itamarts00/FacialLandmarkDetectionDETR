import logging
import math
import os

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import auc

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
    return im


def analyze_results(datastets_inst, datasets, eval_name, output=None, decoder_head=-1, logger=None):
    from main.components.dataclasses import BatchEval, EpochEval
    logger.report_text(f"Analyzing {eval_name.upper()}", logging.INFO, print_console=True)
    datasets = [i.replace("/", "_") for i in datasets]
    full_analysis = list()
    for dataset, dataset_inst in datastets_inst.items():
        if dataset not in datasets:
            continue
        logger.report_text(f"Analysing dataset {dataset} in {eval_name}", logging.INFO, print_console=True)
        # preds, tpts = list(), list()
        epoch_eval = EpochEval(epoch=dataset)
        batch_eval_lst = list()
        for b_idx, b_idx_inst in dataset_inst.items():
            batch_eval = BatchEval(epoch=dataset, batch_idx=b_idx)
            preds, tpts = list(), list()
            [preds.append(b.numpy()) for b in b_idx_inst["preds"]]
            [tpts.append(b.numpy()) for b in b_idx_inst["tpts"]]
            batch_eval.nme, batch_eval.auc08, batch_eval.auc10, _ = evaluate_normalized_mean_error(np.array(preds),
                                                                                                   np.array(tpts))
            batch_eval.end_time()
            batch_eval_lst.append(batch_eval)
        # for i, (tpt, pred) in tqdm(enumerate(zip(tpts, preds))):
        #     preds[i] = min_cost_max_bipartite(pred, tpt)
        epoch_eval.batch_eval_lst = batch_eval_lst
        epoch_eval.end_time()
        logger.report_text(f'Dataset: {dataset} '
                           f'| FR08: {epoch_eval.get_failure_rate(0.08):.3f} '
                           f'| AUC08: {epoch_eval.get_auc(0.08):.3f} '
                           f'| FR10: {epoch_eval.get_failure_rate(0.10):.3f} '
                           f'| AUC10: {epoch_eval.get_auc(0.10):.3f} '
                           f'| NLE: {epoch_eval.get_nle():.3f} ')
        logger.report_scalar(title=f'{dataset}/FR08', series='FR08', value=epoch_eval.get_failure_rate(0.08), iteration=0)
        logger.report_scalar(title=f'{dataset}/AUC08', series='AUC08', value=epoch_eval.get_auc(0.08), iteration=0)
        logger.report_scalar(title=f'{dataset}/FR10', series='AUC10', value=epoch_eval.get_failure_rate(0.10), iteration=0)
        logger.report_scalar(title=f'{dataset}/NLE', series='NLE', value=epoch_eval.get_auc(0.10), iteration=0)
        logger.report_scalar(title=f'{dataset}/FR08', series='FR08', value=epoch_eval.get_nle(), iteration=0)

        full_analysis.append(epoch_eval)
        img_name = f'{eval_name.replace(" ", "_")}-{epoch_eval.epoch}'
        if 'lfpw' in dataset.lower():
            print('here')
        img = save_tough_images(dataset=img_name,
                                dataset_inst=dataset_inst,
                                ds_err=epoch_eval._get_all_values('nme'),
                                output=output,
                                decoder_head=decoder_head)
        logger.report_image(title='Hardest predictions', series=img_name, image=img)

    # TODO: should we check the NME as a weighted sum of the datasets?
    # np.array([i.get_auc(0.08) * (len(i._get_all_values('nme')) / len(tot_err)) for i in full_analysis]).sum()

    tot_err = list()
    [[tot_err.append(i.squeeze()) for i in ds_nme._get_all_values('nme')] for ds_nme in full_analysis]
    tot_err = np.array(tot_err).squeeze()

    failure_rate = lambda nme, rate: (nme > rate).astype(int).mean() * 100
    fail08 = failure_rate(tot_err, 0.08)
    fail10 = failure_rate(tot_err, 0.10)
    auc08 = get_auc(tot_err, thresh=0.08) * 100
    auc10 = get_auc(tot_err, thresh=0.10) * 100
    nle = tot_err.mean() * 100
    logger.report_text(f'Evaluation Set: {eval_name.upper()} '
                       f'| FR08: {fail08:.3f} '
                       f'| AUC08: {auc08:.3f} '
                       f'| FR10: {fail10:.3f} '
                       f'| AUC10: {auc10:.3f} '
                       f'| NLE: {nle:.3f}')

    logger.report_scalar(title=f'{eval_name.upper()}/FR08', series='FR08', value=fail08, iteration=0)
    logger.report_scalar(title=f'{eval_name.upper()}/AUC08', series='AUC08', value=auc08, iteration=0)
    logger.report_scalar(title=f'{eval_name.upper()}/FR10', series='AUC10', value=fail10, iteration=0)
    logger.report_scalar(title=f'{eval_name.upper()}/NLE', series='NLE', value=auc10, iteration=0)
    logger.report_scalar(title=f'{eval_name.upper()}/FR08', series='FR08', value=nle, iteration=0)


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


def rearrange_prediction_for_min_cos_max_bipartite(prediction, gt):
    gt_np = gt.detach().cpu().numpy() if not isinstance(gt, np.ndarray) else gt
    prediction_np = prediction.detach().cpu().numpy() if not isinstance(prediction, np.ndarray) else prediction
    for i, (pred, gtpt) in enumerate(zip(prediction_np, gt_np)):
        prediction_np[i] = min_cost_max_bipartite(pred, gtpt)
    return prediction_np


def min_cost_max_bipartite(prediction, gt):
    init = init_cost_matrix(prediction, gt)
    tpts_idx, preds_idx = linear_sum_assignment(init, maximize=False)
    new_preds = np.array([prediction[i] for i in preds_idx])
    return new_preds


def init_cost_matrix(prediction, gt):
    point_nme = lambda pt1, pt2: np.linalg.norm(pt1 - pt2)
    init = np.zeros([gt.shape[0], gt.shape[0]])
    for row_idx, row in enumerate(init):
        for col_idx, ele in enumerate(row):
            init[col_idx][row_idx] = point_nme(prediction[row_idx], gt[col_idx])
    return init
