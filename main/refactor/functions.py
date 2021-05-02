# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import time

import torch
import wandb

from main.components.hm_regression import *
from main.refactor.evaluation_functions import decode_preds, compute_nme
from utils.plot_utils import plot_gt_pred_on_img

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(train_loader, model, criterion, optimizer,
                epoch, writer_dict, multi_dec_loss=False, **kwargs):
    max_norm = 0
    log_interval = kwargs.get('log_interval', 20)
    debug = kwargs.get('debug', False)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    [criteria.train() for criteria in criterion.values()]

    nme_count = nme_batch_sum = 0

    end = time.time()

    for i, item in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)

        input_, target, opts = item['img'], item['target'].cuda(), item['opts'].cuda()
        scale, hm_factor, heatmaps = item['sfactor'].cuda(), item['hmfactor'], item['heatmaps'].cuda()
        weighted_loss_mask_awing = item['weighted_loss_mask_awing'].cuda()
        # compute the output
        input_ = input_.cuda()
        bs = target.shape[0]

        target_dict = {'labels': [torch.range(start=0, end=target.shape[1] - 1) for i in range(bs)],
                       'coords': target}

        output, hm_encoder = model(input_)

        # Loss
        loss_dict = criterion['coord_loss_criterion'](output, target_dict)
        coords_dec_loss = loss_dict['coords']

        lossv = sum(coords_dec_loss) if multi_dec_loss else coords_dec_loss[-1]

        hm_backbone_regression = True if output['hm_output'] is not None else False
        if hm_backbone_regression:
            hm_loss = criterion['hm_regression_criterion'](output['hm_output'], heatmaps, M=weighted_loss_mask_awing)
            lossv = lossv.add(hm_loss)

        multi_enc_loss = not all([i is None for i in hm_encoder])
        if multi_enc_loss:
            enc_loss = torch.stack(
                [criterion['enc_loss_criterion'](hm, heatmaps, M=weighted_loss_mask_awing) for hm in hm_encoder])
            tot_enc_loss = torch.sum(enc_loss)
            lossv = lossv.add_(tot_enc_loss) if multi_enc_loss else lossv

        if not math.isfinite(lossv.item()):
            print("Loss is {}, stopping training".format(lossv.item()))
            sys.exit(1)

        # NME
        preds = output['pred_coords'][-1] * 255
        scale_matrix = scale[:, np.newaxis, np.newaxis] * torch.ones_like(preds)
        opts_scaled = opts * scale_matrix
        nme_batch = compute_nme(preds, opts_scaled)
        nme_batch_sum += nme_batch.sum()
        nme_count += preds.shape[0]

        # optimize
        optimizer.zero_grad()
        lossv.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        losses.update(lossv.item(), input_.size(0))

        batch_time.update(time.time() - end)
        if i % log_interval == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input_.size(0) / batch_time.val,
                data_time=data_time, loss=losses)
            logger.info(msg)

        if debug:
            break

        end = time.time()

    nme = torch.true_divide(nme_batch_sum, nme_count)
    wandb.log({'train/nme': nme, 'epoch': epoch})
    wandb.log({'train/loss': losses.avg, 'epoch': epoch})
    wandb.log({'train/batch_time': batch_time.avg, 'epoch': epoch})

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
        writer_dict['train_global_steps'] = global_steps + 1

    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f}' \
        .format(epoch, batch_time.avg, losses.avg, nme)
    logger.info(msg)


def validate_epoch(val_loader, model, criterion, epoch, writer_dict, multi_dec_loss, **kwargs):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = kwargs.get('num_landmarks', 20)
    debug = kwargs.get('debug', False)

    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()
    [criteria.eval() for criteria in criterion.values()]

    nme_count = nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, item in enumerate(val_loader):
            data_time.update(time.time() - end)
            input_, target, opts = item['img'], item['target'].cuda(), item['opts'].cuda()
            scale, hm_factor, heatmaps = item['sfactor'].cuda(), item['hmfactor'], item['heatmaps'].cuda()
            weighted_loss_mask_awing = item['weighted_loss_mask_awing'].cuda()

            # compute the output
            input_ = input_.cuda()
            bs = target.shape[0]

            target_dict = {'labels': [torch.range(start=0, end=target.shape[1] - 1) for i in range(bs)],
                           'coords': target.cuda()}
            output, hm_encoder = model(input_)

            # loss
            loss_dict = criterion['coord_loss_criterion'](output, target_dict)
            coords_dec_loss = loss_dict['coords']
            lossv = sum(coords_dec_loss) if multi_dec_loss else coords_dec_loss[-1]

            hm_backbone_regression = True if output['hm_output'] is not None else False
            if hm_backbone_regression:
                hm_loss = criterion['hm_regression_criterion'](output['hm_output'], heatmaps,
                                                               M=weighted_loss_mask_awing)
                lossv = lossv.add(hm_loss)
                hm_regression_loss = hm_loss

            multi_enc_loss = not all([i is None for i in hm_encoder])
            if multi_enc_loss:
                enc_loss = torch.stack(
                    [criterion['enc_loss_criterion'](hm, heatmaps, M=weighted_loss_mask_awing) for hm in hm_encoder])
                tot_enc_loss = torch.sum(enc_loss)
                lossv = lossv.add_(tot_enc_loss) if multi_enc_loss else lossv

            # NME
            preds = output['pred_coords'][-1] * 256
            scale_matrix = scale[:, np.newaxis, np.newaxis] * torch.ones_like(preds)
            opts_scaled = opts * scale_matrix
            nme_batch = compute_nme(preds, opts_scaled)
            nme_batch_sum += nme_batch.sum()
            nme_count += preds.shape[0]

            # scatter_prediction_gt(preds, opts)

            # Failure Rate under different threshold
            failure_008 = (nme_batch > 0.08).sum()
            failure_010 = (nme_batch > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            losses.update(lossv.item(), input_.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if debug:
                break

    nme = torch.true_divide(nme_batch_sum, nme_count)
    failure_008_rate = torch.true_divide(count_failure_008, nme_count)
    failure_010_rate = torch.true_divide(count_failure_010, nme_count)

    msg = 'Test Epoch {} time: {:.4f} loss:{:.4f} nme: {:.4f} [008]: {:.4f} ' \
          '[010]: {:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                 failure_008_rate, failure_010_rate)
    logger.info(msg)

    dbg_img = plot_gt_pred_on_img(item=item, predictions=preds, index=-1)
    grid = torch.tensor(np.swapaxes(np.swapaxes(dbg_img, 0, -1), 1, 2))

    wandb.log({'valid/nme': nme, 'epoch': epoch})
    wandb.log({'valid/loss': losses.avg, 'epoch': epoch})
    wandb.log({'valid/fail_rate_008': failure_008_rate, 'epoch': epoch})
    wandb.log({'valid/fail_rate_010': failure_010_rate, 'epoch': epoch})
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
        [log[epoch].update({k: v}) for (k, v) in loss_dict.items()]
        [writer.add_scalar(f'loss_coords_dec_{i}', v, global_steps) for (i, v) in enumerate(loss_dict['coords'])]
        if multi_enc_loss:
            [writer.add_scalar(f'loss_hm_enc_{i}', v, global_steps) for (i, v) in enumerate(enc_loss)]
        if hm_backbone_regression:
            log[epoch].update({'hm_backbone_regression': hm_loss})
        writer.add_image('images', grid, global_steps)
        log[epoch].update({'dbg_img': dbg_img})
        writer_dict['valid_global_steps'] = global_steps + 1
    return nme, predictions


def inference(model, data_loader, **kwargs):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = kwargs.get('num_classes', 68)
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, item in enumerate(data_loader):
            data_time.update(time.time() - end)
            input_, target = item['img'], item['target']
            output = model(input_)

            scale = item['sfactor']
            score_map = output.data.cpu()
            res = np.array(item['target'].shape[-2:])
            center = np.zeros([score_map.shape[0], 2])
            preds = decode_preds(score_map, center, scale, res)
            opts = item['opts']

            # NME
            nme_temp = compute_nme(preds.numpy(), opts.cpu().numpy())

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[item['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    return nme, predictions


def single_image_train(train_loader, model, criterion, optimizer, epochs, writer_dict, **kwargs):
    log_interval = kwargs.get('log_interval', 20)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    criterion['coord_loss_criterion'].train()
    max_norm = 0
    nme_count = nme_batch_sum = 0
    end = time.time()

    item = next(iter(train_loader))
    data_time.update(time.time() - end)

    input_, target, opts = item['img'], item['target'], item['opts']
    scale, hm_factor = item['sfactor'], item['hmfactor']

    # compute the output
    input_ = input_.cuda()
    bs = target.shape[0]
    # target = torch.cat((target, 16 * torch.ones_like(target)), dim=2)
    target_dict = [{'labels': torch.range(start=0, end=target.shape[1] - 1).cuda(),
                    'coords': target[i].cuda()} for i in range(bs)]

    for epoch in range(0, epochs + 1):

        output = model(input_)
        # Loss
        loss_dict = criterion['coord_loss_criterion'](output, target_dict)
        weight_dict = criterion['coord_loss_criterion'].weight_dict
        lossv = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if not math.isfinite(lossv.item()):
            print("Loss is {}, stopping training".format(lossv.item()))
            sys.exit(1)

        # NME
        preds = output['pred_coords'].cpu().detach().numpy()[-1] * 256
        nme_batch = compute_nme(preds, opts.cpu().numpy())
        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
        nme_count = nme_count + preds.shape[0]

        # optimize
        optimizer.zero_grad()
        lossv.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        losses.update(lossv.item(), input_.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        nme = nme_batch_sum / nme_count

        dbg_img = plot_gt_pred_on_img(item=item, predictions=preds, index=-1)
        grid = torch.tensor(np.swapaxes(np.swapaxes(dbg_img, 0, -1), 1, 2))

        if writer_dict:
            writer = writer_dict['writer']
            log = writer_dict['log']
            log[epoch] = {}
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            log[epoch].update({'train_loss': losses.val})
            writer.add_scalar('train_nme', nme, global_steps)
            log[epoch].update({'train_nme': nme})
            writer.add_scalar('batch_time.avg', batch_time.avg, global_steps)
            log[epoch].update({'batch_time.avg': batch_time.avg})
            [log[epoch].update({k: v}) for (k, v) in loss_dict.items()]
            [writer.add_scalar(k, v, global_steps) for (k, v) in loss_dict.items()]
            writer.add_image('images', grid, global_steps)
            log[epoch].update({'dbg_img': dbg_img})
            writer_dict['train_global_steps'] = global_steps + 1
        msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f}' \
            .format(epoch, batch_time.avg, losses.avg, nme)
        logger.info(msg)
