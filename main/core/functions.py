from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import sys
import time

import numpy as np
import torch
import wandb

from main.components.ptsutils import decode_preds_heatmaps
from main.core.evaluation_functions import evaluate_normalized_mean_error, get_auc, \
    rearrange_prediction_for_min_cos_max_bipartite
from utils.data_organizer import AverageMeter
from utils.plot_utils import plot_gt_pred_on_img
from main.core.evaluation_functions import min_cost_max_bipartite
logger = logging.getLogger(__name__)


def train_epoch(train_loader, model, criteria, optimizer, scheduler, epoch, writer_dict, **kwargs):
    max_norm = 0
    log_interval = kwargs.get('log_interval', 20)
    debug = kwargs.get('debug', False)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    scheduler.step()
    criteria.train()

    nme_vec = list()
    end = time.time()

    for i, item in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)

        input_, tpts = item['img'].cuda(), item['tpts'].cuda()
        scale, hm_factor, heatmaps = item['sfactor'].cuda(), item['hmfactor'], item['heatmaps'].cuda()
        weighted_loss_mask_awing = item['weighted_loss_mask_awing'].cuda()
        # compute the output
        bs = tpts.shape[0]

        target_dict = {'labels': [torch.range(start=0, end=tpts.shape[1] - 1) for i in range(bs)],
                       'coords': tpts, 'heatmap_bb': heatmaps,
                       'weighted_loss_mask_awing': weighted_loss_mask_awing}

        output, preds = inference(model, input_batch=input_, **kwargs)
        preds_new = rearrange_prediction_for_min_cos_max_bipartite(preds, tpts).cuda()
        loss_dict, lossv = get_loss(criteria, output, target_dict=target_dict, **kwargs)

        if not math.isfinite(lossv.item()):
            print("Loss is {}, stopping training".format(lossv.item()))
            sys.exit(1)

        # NME
        nme_batch, auc08_batch, auc10_batch, for_pck_curve_batch = evaluate_normalized_mean_error(preds, tpts)
        nme_vec.append(nme_batch)

        # optimize
        optimizer.zero_grad()
        lossv.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        losses.update(lossv.item(), input_.size(0))

        batch_time.update(time.time() - end)
        if i % log_interval == 0:
            speed = str(int(input_.size(0) / batch_time.val)).zfill(3)
            msg = f'Epoch: [{str(epoch).zfill(3)}][{str(i).zfill(3)}/{len(train_loader)}]\t' \
                  f' Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t Speed: {speed}' \
                  f' samples/s\t Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  f' Loss: {losses.val:.5f}\t AUC08: {auc08_batch:.2f}\t'
            logger.info(msg)

        if debug:
            break

        end = time.time()

    ls = []
    [[ls.append(i) for i in j] for j in nme_vec]
    nme = np.array(ls).mean()
    failure_008_rate = (np.hstack(np.array(ls).squeeze()) > 0.08).astype(int).mean()
    failure_010_rate = (np.hstack(np.array(ls).squeeze()) > 0.10).astype(int).mean()
    auc08_epoch = get_auc(np.array(ls), 0.08) * 100
    auc10_epoch = get_auc(np.array(ls), 0.10) * 100
    wandb.log({'train/nme': nme, 'epoch': epoch})
    wandb.log({'train/loss': losses.avg, 'epoch': epoch})
    wandb.log({'train/failure_008_rate': failure_008_rate, 'epoch': epoch})
    wandb.log({'train/failure_010_rate': failure_010_rate, 'epoch': epoch})
    wandb.log({'train/auc08_epoch': auc08_epoch, 'epoch': epoch})
    wandb.log({'train/auc10_epoch': auc10_epoch, 'epoch': epoch})

    if writer_dict:
        writer = writer_dict['writer']
        log = writer_dict['log']
        log[epoch] = {}
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', losses.avg, global_steps)
        log[epoch].update({'train_loss': losses.avg})
        writer.add_scalar('train_nme', nme, global_steps)
        log[epoch].update({'train_nme': nme})
        writer.add_scalar('batch_time.avg', batch_time.avg, global_steps)
        log[epoch].update({'batch_time.avg': batch_time.avg})
        writer.add_scalar('failure_008_rate', failure_008_rate, global_steps)
        log[epoch].update({'failure_008_rate': failure_008_rate})
        writer.add_scalar('failure_010_rate', failure_010_rate, global_steps)
        log[epoch].update({'failure_010_rate': failure_010_rate})
        writer.add_scalar('auc08_epoch', auc08_epoch, global_steps)
        log[epoch].update({'auc08_epoch': auc08_epoch})
        writer.add_scalar('auc10_epoch', auc10_epoch, global_steps)
        log[epoch].update({'auc10_epoch': auc10_epoch})

        writer_dict['train_global_steps'] = global_steps + 1

    msg = f'Train Epoch {epoch} | time: {batch_time.avg:.4f} | loss: {losses.avg:.4f} | NME: {nme:.4f}' \
          f' | AUC08: {auc08_epoch:.3f} | FR08 : {failure_008_rate:.3f}'
    logger.info(msg)


def validate_epoch(val_loader, model, criteria, epoch, writer_dict, **kwargs):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    debug = kwargs.get('debug', False)

    model.eval()
    criteria.eval()

    nme_vec = list()
    end = time.time()

    with torch.no_grad():
        for i, item in enumerate(val_loader):
            data_time.update(time.time() - end)
            input_, tpts = item['img'].cuda(), item['tpts'].cuda()
            scale, hm_factor, heatmaps = item['sfactor'].cuda(), item['hmfactor'], item['heatmaps'].cuda()
            weighted_loss_mask_awing = item['weighted_loss_mask_awing'].cuda()

            bs = tpts.shape[0]

            target_dict = {'labels': [torch.range(start=0, end=tpts.shape[1] - 1) for i in range(bs)],
                           'coords': tpts, 'heatmap_bb': heatmaps,
                           'weighted_loss_mask_awing': weighted_loss_mask_awing}

            output, preds = inference(model, input_batch=input_, **kwargs)
            loss_dict, lossv = get_loss(criteria, output, target_dict=target_dict, **kwargs)

            # NME

            nme_batch, auc08_batch, auc10_batch, for_pck_curve_batch = evaluate_normalized_mean_error(preds, tpts)
            nme_vec.append(nme_batch)

            losses.update(lossv.item(), input_.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if debug:
                break

    ls = []
    [[ls.append(i) for i in j] for j in nme_vec]
    nme_vec_np = np.array(ls)
    nme = nme_vec_np.mean()
    failure_008_rate = (np.hstack(nme_vec_np.squeeze()) > 0.08).astype(int).mean()
    failure_010_rate = (np.hstack(nme_vec_np.squeeze()) > 0.10).astype(int).mean()
    auc08_epoch = get_auc(nme_vec_np, 0.08) * 100
    auc10_epoch = get_auc(nme_vec_np, 0.10) * 100

    msg = f'Test Epoch {epoch}  | time: {batch_time.avg:.4f} | loss:{losses.avg:.4f}  | NME: {nme:.4f}' \
          f' | AUC08: {auc08_epoch:.3f} | FR08: {failure_008_rate:.3f}'
    logger.info(msg)

    dbg_img = plot_gt_pred_on_img(item=item, predictions=preds, index=0)
    grid = torch.tensor(np.swapaxes(np.swapaxes(dbg_img, 0, -1), 1, 2))

    wandb.log({'valid/nme': nme, 'epoch': epoch})
    wandb.log({'valid/loss': losses.avg, 'epoch': epoch})
    wandb.log({'valid/fail_rate_008': failure_008_rate, 'epoch': epoch})
    wandb.log({'valid/fail_rate_010': failure_010_rate, 'epoch': epoch})
    wandb.log({'valid/auc08': auc08_epoch, 'epoch': epoch})
    wandb.log({'valid/auc10': auc10_epoch, 'epoch': epoch})
    wandb.log({'valid/batch_time': batch_time.avg, 'epoch': epoch})
    # wandb.log({"debug_image": dbg_img})

    if writer_dict:
        writer = writer_dict['writer']
        log = writer_dict['log']
        log[epoch] = {}
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        log[epoch].update({'valid_loss': losses.avg})
        writer.add_scalar('valid_nme', nme, global_steps)
        log[epoch].update({'valid_nme': nme})
        writer.add_scalar('valid_failure_008_rate', failure_008_rate, global_steps)
        log[epoch].update({'valid_failure_008_rate': failure_008_rate})
        writer.add_scalar('valid_failure_010_rate', failure_010_rate, global_steps)
        log[epoch].update({'valid_failure_010_rate': failure_010_rate})
        writer.add_scalar('valid_auc010', auc10_epoch, global_steps)
        log[epoch].update({'valid_auc010': auc10_epoch})
        writer.add_scalar('valid_auc08', auc08_epoch, global_steps)
        log[epoch].update({'valid_auc08': auc08_epoch})
        [log[epoch].update({k: v}) for (k, v) in loss_dict.items()]
        [writer.add_scalar(k, v, global_steps) for k, v in loss_dict.items()]
        writer.add_image('images', grid, global_steps)
        log[epoch].update({'dbg_img': dbg_img})
        writer_dict['valid_global_steps'] = global_steps + 1
    return nme


def inference(model, input_batch, **kwargs):
    # inference
    model_name = kwargs.get('model_name', None)
    output_ = model(input_batch)

    if model_name == 'HRNET':
        preds = decode_preds_heatmaps(output_).cuda()
    if model_name == 'DETR':
        decoder_head = kwargs.get('decoder_head', -1)
        preds = output_['pred_coords'][decoder_head] # +0.5 from HRNET
    return output_, preds


def get_loss(criteria, output, target_dict, **kwargs):
    model_name = kwargs.get('model_name', None)
    # Loss
    if model_name == 'HRNET':
        hm_amp_factor = kwargs.get('hm_amp_factor', 1)
        heatmaps = target_dict['heatmap_bb']
        lossv = criteria(output, heatmaps * hm_amp_factor)
        loss_dict = {'MSE_loss': lossv.item()}
    if model_name == 'DETR':
        loss_dict, lossv = criteria(output, target_dict)
    return loss_dict, lossv
