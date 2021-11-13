# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class IMGJoints_MSELoss(nn.Module):
    def __init__(self):
        super(IMGJoints_MSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)
        loss = 0
        for idx in range(batch_size):
            for joint in range(num_joints):
                loss += self.criterion(output[idx][joint], target[idx][joint])
        return loss


class JointsMSELoss(nn.Module):
    def __init__(self):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, output, target):
        num_joints = output.size(1)
        loss = 0
        for joint in range(num_joints):
            loss += self.criterion(output[:, joint], target[:, joint])
        return loss


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            loss.append(0.5 * self.criterion(heatmap_pred, heatmap_gt))

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)


class EncL1_Loss(nn.Module):
    def __init__(self):
        super(EncL1_Loss, self).__init__()
        self.l2criterion = nn.MSELoss(reduction='mean')
        self.l1criterion = nn.L1Loss(reduction='mean')

    def forward(self, output, target_dict):
        l1 = self.l2criterion(output[0], target_dict['coords'])
        l2 = self.l2criterion(output[1], target_dict['distance_matrix'])
        return l1 + l2