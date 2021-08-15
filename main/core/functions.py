from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import sys

import torch

from main.components.misc_dataclasses import EpochEval, BatchEval
from main.components.ptsutils import decode_preds_heatmaps
from main.core.evaluation_functions import evaluate_normalized_mean_error
from utils.plot_utils import plot_gt_pred_on_img


def train_epoch(train_loader, model, criteria, optimizer, scheduler, epoch, logger_cml, **kwargs):
    max_norm = 0
    log_interval = kwargs.get('log_interval', 20)
    debug = kwargs.get('debug', False)
    train_with_heatmaps = kwargs.get('train_with_heatmaps', False)

    epoch_eval = EpochEval(epoch=epoch)
    batch_eval_lst = list()

    model.train()
    scheduler.step()
    criteria.train()

    for batch_idx, item in enumerate(train_loader):
        # measure data time
        batch_eval = BatchEval(epoch=epoch, batch_idx=batch_idx)
        input_, tpts, scale = item['img'].cuda(), item['tpts'].cuda(), item['sfactor'].cuda()
        # compute the output
        bs = tpts.shape[0]
        target_dict = {'labels': [torch.range(start=0, end=tpts.shape[1] - 1) for i in range(bs)], 'coords': tpts, }

        if train_with_heatmaps:
            hm_factor, heatmaps, weighted_loss_mask_awing = item['hmfactor'], item['heatmaps'].cuda(), \
                                                            item['weighted_loss_mask_awing'].cuda()
            target_dict.update({'heatmap_bb': heatmaps, 'weighted_loss_mask_awing': weighted_loss_mask_awing})

        output, preds = inference(model, input_batch=input_, **kwargs)
        # preds = rearrange_prediction_for_min_cos_max_bipartite(preds, tpts)
        loss_dict, lossv = get_loss(criteria, output, target_dict=target_dict, **kwargs)

        if not math.isfinite(lossv.item()):
            logger_cml.report_text(f"Loss is {lossv.item()}, stopping training", level=logging.INFO, print_console=True)
            sys.exit(1)

        # NME
        batch_eval.nme, batch_eval.auc08, batch_eval.auc10, for_pck_curve_batch = evaluate_normalized_mean_error(preds,
                                                                                                                 tpts)

        # optimize
        optimizer.zero_grad()
        lossv.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        batch_eval.loss = lossv.item()
        batch_eval.end_time()
        batch_eval_lst.append(batch_eval)

        # Messages
        if batch_idx % log_interval == 0:
            msg = f'Epoch: [{str(epoch).zfill(3)}][{str(batch_idx).zfill(3)}/{len(train_loader)}]\t ' \
                  f'| Time: {batch_eval.process_time:.3f}s \t ' \
                  f'| Speed: {batch_eval.sample_per_sec():.2f} sample/sec \t ' \
                  f'| Loss: {batch_eval.loss:.5f}\t ' \
                  f'| AUC08: {batch_eval.auc08:.2f}\t'
            logger_cml.report_text(msg, level=logging.INFO, print_console=True)

        if debug:
            break

    epoch_eval.batch_eval_lst = batch_eval_lst
    epoch_eval.end_time()

    logger_cml.report_scalar('train/nme', 'nme', value=epoch_eval.nme_avg(), iteration=epoch)
    logger_cml.report_scalar('train/loss', 'loss', value=epoch_eval.loss_avg(), iteration=epoch)
    logger_cml.report_scalar('train/failure_008_rate', 'failure_008_rate', value=epoch_eval.get_failure_rate(0.08),
                             iteration=epoch)
    logger_cml.report_scalar('train/failure_010_rate', 'failure_010_rate', value=epoch_eval.get_failure_rate(0.10),
                             iteration=epoch)
    logger_cml.report_scalar('train/auc08', 'auc08', value=epoch_eval.get_auc(0.08), iteration=epoch)
    logger_cml.report_scalar('train/auc10', 'auc10', value=epoch_eval.get_auc(0.10), iteration=epoch)

    msg = f'Train Epoch {epoch} ' \
          f'| process time: {epoch_eval.process_time():.4f} sec' \
          f'| batch_avg_time: {epoch_eval.average_process_time():.4f} sec' \
          f'| loss: {epoch_eval.loss_avg():.4f} ' \
          f'| NME: {epoch_eval.nme_avg():.4f} ' \
          f'| AUC08: {epoch_eval.get_auc(0.08):.3f} ' \
          f'| FR08 : {epoch_eval.get_failure_rate(0.08):.3f}'
    logger_cml.report_text(msg, level=logging.INFO, print_console=True)


def validate_epoch(val_loader, model, criteria, epoch, logger_cml, **kwargs):
    debug = kwargs.get('debug', False)
    train_with_heatmaps = kwargs.get('train_with_heatmaps', False)

    epoch_eval = EpochEval(epoch=epoch)
    batch_eval_lst = list()

    model.eval()
    criteria.eval()

    with torch.no_grad():
        for batch_idx, item in enumerate(val_loader):
            # measure data time
            batch_eval = BatchEval(epoch=epoch, batch_idx=batch_idx)
            input_, tpts, scale = item['img'].cuda(), item['tpts'].cuda(), item['sfactor'].cuda()
            # compute the output
            bs = tpts.shape[0]
            target_dict = {'labels': [torch.range(start=0, end=tpts.shape[1] - 1) for i in range(bs)], 'coords': tpts, }

            if train_with_heatmaps:
                hm_factor, heatmaps, weighted_loss_mask_awing = item['hmfactor'], item['heatmaps'].cuda(), \
                                                                item['weighted_loss_mask_awing'].cuda()
                target_dict.update({'heatmap_bb': heatmaps, 'weighted_loss_mask_awing': weighted_loss_mask_awing})

            output, preds = inference(model, input_batch=input_, **kwargs)
            # preds = rearrange_prediction_for_min_cos_max_bipartite(preds, tpts)
            loss_dict, lossv = get_loss(criteria, output, target_dict=target_dict, **kwargs)

            # NME
            batch_eval.nme, batch_eval.auc08, batch_eval.auc10, for_pck_curve_batch = evaluate_normalized_mean_error(
                preds, tpts)

            batch_eval.loss = lossv.item()
            batch_eval.end_time()
            batch_eval_lst.append(batch_eval)

            if debug:
                break

    epoch_eval.batch_eval_lst = batch_eval_lst
    epoch_eval.end_time()
    logger_cml.report_scalar('test/nme', 'nme', value=epoch_eval.nme_avg(), iteration=epoch)
    logger_cml.report_scalar('test/loss', 'loss', value=epoch_eval.loss_avg(), iteration=epoch)
    logger_cml.report_scalar('test/failure_008_rate', 'failure_008_rate', value=epoch_eval.get_failure_rate(0.08),
                             iteration=epoch)
    logger_cml.report_scalar('test/failure_010_rate', 'failure_010_rate', value=epoch_eval.get_failure_rate(0.10),
                             iteration=epoch)
    logger_cml.report_scalar('test/auc08', 'auc08', value=epoch_eval.get_auc(0.08), iteration=epoch)
    logger_cml.report_scalar('test/auc10', 'auc10', value=epoch_eval.get_auc(0.10), iteration=epoch)

    msg = f'Test Epoch {epoch}  ' \
          f'| time: {epoch_eval.average_process_time():.4f} sec' \
          f'| loss:{epoch_eval.loss_avg():.4f} ' \
          f'| NME: {epoch_eval.nme_avg():.4f} ' \
          f'| AUC08: {epoch_eval.get_auc(0.08):.3f} ' \
          f'| FR08: {epoch_eval.get_failure_rate(0.08):.3f}'
    logger_cml.report_text(msg, level=logging.INFO, print_console=True)

    dbg_img = plot_gt_pred_on_img(item=item, predictions=preds, index=0)
    logger_cml.report_image('debug_image', 'converging landmarks', iteration=epoch, image=dbg_img)

    return epoch_eval.nme_avg()


def inference(model, input_batch, **kwargs):
    # inference
    model_name = kwargs.get('model_name', None)
    output_ = model(input_batch)

    if model_name == 'PERC':
        preds = output_
    if model_name == 'HRNET':
        preds = decode_preds_heatmaps(output_).cuda()
    if model_name == 'DETR':
        decoder_head = kwargs.get('decoder_head', -1)
        preds = output_['pred_coords'][decoder_head]  # +0.5 from HRNET
    return output_, preds


def get_loss(criteria, output, target_dict, **kwargs):
    model_name = kwargs.get('model_name', None)
    # Loss
    if model_name == 'HRNET':
        hm_amp_factor = kwargs.get('hm_amp_factor', 1)
        heatmaps = target_dict['heatmap_bb']
        lossv = criteria(output, heatmaps * hm_amp_factor)
        loss_dict = {'MSE_loss': lossv.item()}
    elif model_name == 'DETR':
        loss_dict, lossv = criteria(output, target_dict)
    elif model_name == 'PERC':
        lossv = criteria(output, target_dict['coords'])
        loss_dict = None
    return loss_dict, lossv
