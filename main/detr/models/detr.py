# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from main.components.Awing.awing_loss import Loss_weighted
from main.detr.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    get_world_size,
    is_dist_avail_and_initialized,
)
from .backbone import build_backbone
from .transformer import build_transformer


class DETR(nn.Module):
    """This is the DETR module that performs object detection"""

    def __init__(self, backbone, transformer, num_classes, num_queries):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, samples: NestedTensor):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_coords": The normalized coords coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the un-normalized bounding box.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos, hm_reg = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs, memory = self.transformer(
            self.input_proj(src), mask, self.query_embed.weight, pos[-1]
        )

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid() * 255
        out = {
            "pred_logits": outputs_class,
            "pred_coords": outputs_coord,
            "hm_output": hm_reg,
        }

        return out


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth coords and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
            self,
            last_dec_coord_loss=True,
            multi_dec_loss=False,
            multi_enc_loss=False,
            heatmap_regression_via_backbone=False,
    ):
        """Create the criterion.
        Parameters:
            l1_coord_loss: flag, L1 loss from the last decoder output
            multi_dec_loss: flag, sum of L1 loss from all decoder outputs
            multi_enc_loss: HM regression from the output of all encoders
            hm_regression_via_bb: flag, HM regression from the backbone intermediate layer
        """
        super().__init__()
        self.last_dec_coord_loss_flag = last_dec_coord_loss
        self.multi_dec_loss_flag = multi_dec_loss
        self.multi_enc_loss_flag = multi_enc_loss
        self.hm_regression_via_bb_flag = heatmap_regression_via_backbone
        self.losses = self.get_loss_map()
        self.awing_loss = self.load_awing_loss()

    def load_awing_loss(self):
        if self.hm_regression_via_bb_flag or self.multi_enc_loss_flag:
            return Loss_weighted()
        else:
            return None

    @staticmethod
    def l1_coord_loss(outputs, targets, num_coords):
        return F.l1_loss(outputs, targets, reduction="mean")

    @staticmethod
    def l2_coord_loss(outputs, targets, num_coords):
        return F.mse_loss(outputs, targets, reduction="mean")

    def last_dec_coord_loss(self, outputs, targets, num_coords, loss_type="l2"):
        preds = outputs["pred_coords"][-1]
        opts = targets["coords"]
        loss = self.l2_coord_loss if loss_type == "l2" else self.l1_coord_loss
        res = loss(outputs=preds, targets=opts, num_coords=num_coords)
        return res

    def multi_dec_loss(self, outputs, targets, num_coords, type="l2"):
        preds = outputs["pred_coords"]
        opts = targets["coords"]
        loss = self.l2_coord_loss if type == "l2" else self.l1_coord_loss
        res = [
            loss(outputs=dec_head, targets=opts, num_coords=num_coords)
            for dec_head in preds
        ]
        return sum(res)

    def multi_enc_loss(self, outputs, targets, num_coords):
        raise NotImplementedError()

    def hm_regression_via_bb(self, outputs, targets, num_coords):
        heatmaps_opts = targets["heatmap_bb"]
        heatmaps_pred = outputs["hm_output"]
        weighted_loss_mask_awing = targets["weighted_loss_mask_awing"]
        loss = self.awing_loss(heatmaps_pred, heatmaps_opts, M=weighted_loss_mask_awing)
        return loss

    def get_loss_map(self, **kwargs):
        loss_map = dict()
        if self.last_dec_coord_loss_flag and not self.multi_dec_loss_flag:
            loss_map.update({"last_dec_coord_loss": self.last_dec_coord_loss})
        if self.multi_dec_loss_flag:
            loss_map.update({"multi_dec_loss": self.multi_dec_loss})
        if self.multi_enc_loss_flag:
            loss_map.update({"multi_enc_loss": self.multi_enc_loss})
        if self.hm_regression_via_bb_flag:
            loss_map.update({"hm_regression_via_bb": self.hm_regression_via_bb})
        return loss_map

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Compute the average number of target coords across all nodes, for normalization purposes
        num_coords = sum([len(t) for t in targets["labels"]])
        num_coords = torch.as_tensor(
            [num_coords], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_coords)
        num_coords = torch.clamp(num_coords / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss_name, loss_func in self.losses.items():
            losses.update({loss_name: loss_func(outputs, targets, num_coords)})
        lossv = sum(list(losses.values()))
        return losses, lossv


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def load_criteria(args):
    device = torch.device(args.device)
    criterion = SetCriterion(
        last_dec_coord_loss=args.last_dec_coord_loss,
        multi_dec_loss=args.multi_dec_loss,
        multi_enc_loss=args.multi_enc_loss,
        heatmap_regression_via_backbone=args.heatmap_regression_via_backbone,
    )
    criterion.to(device)
    return criterion


def build(args):
    num_classes = args.num_classes

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
    )
    # for param in model.parameters():
    #     param.requires_grad = True

    return model

# class MultiTaskLoss(nn.Module):
#     def __init__(self, tasks):
#         super(MultiTaskLoss, self).__init__()
#         self.tasks = nn.ModuleList(tasks)
#         self.sigma = nn.Parameter(torch.ones(len(tasks)))
#         self.mse = nn.MSELoss()
#
#     def forward(self, x, targets):
#        l = [self.mse(f(x), y) for y, f in zip(targets, self.tasks)]
#        l = 0.5 * torch.Tensor(l) / self.sigma**2
#        l = l.sum() + torch.log(self.sigma.prod())
#        return l
