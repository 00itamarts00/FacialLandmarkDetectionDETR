from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path as osp
import torch

import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
import cv2

from main.core.functions import inference
from utils.plot_utils import plot_grid_of_ldm


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def inspect_atten_map_by_locations(item, model, model_name="transpose", mode="dependency", threshold=None, index=0,
                                   device=torch.device("cuda"), kpt_color="white", img_name="image", save_img=False,
                                   **kwargs):
    """
    Visualize the attention maps in all of attention layers
    Args:
        image: shape -> [3, h, w]; type -> torch.Tensor;
        model: a pretrained model; type -> torch.nn.Module
        query_locations: shape -> [K,2]: type -> np.array
        mode: 'dependency' or 'affect'
        threshold: Default: None. If using it, recommend to be 0.01
    """
    assert mode in ["dependency", "affect"]
    input_, tpts = item['img'].cuda(), item['tpts'].cuda()

    # inputs = torch.cat([image.to(device)]).unsqueeze(0)
    # images = torch.cat([image.to(device)]).unsqueeze(0).detach().cpu().numpy()
    # img_vis_kpts = plot_grid_of_ldm('None', images, np.expand_dims(query_locations, 0),
    #                                 np.expand_dims(query_locations, 0), s=5)

    features = []
    global_enc_atten_maps = []

    feature_hooks = [model.reduce.register_forward_hook(lambda self, input, output1: features.append(output1))]

    atten_maps_hooks = [model.global_encoder.layers[i].self_attn.register_forward_hook(
        lambda self, input, output: global_enc_atten_maps.append(output[1])) for i in
        range(len(model.global_encoder.layers))]

    with torch.no_grad():
        output, preds_tta = inference(model=model, input_batch=input_, **kwargs)
        for h in feature_hooks:
            h.remove()
        for h in atten_maps_hooks:
            h.remove()

    shape = features[0].shape[-2:]
    enc_atten_maps_hwhw = []
    for atten_map in global_enc_atten_maps:
        atten_map = atten_map.reshape(shape + shape)
        enc_atten_maps_hwhw.append(atten_map)

    attn_layers_num = len(enc_atten_maps_hwhw)
    down_rate = img_vis_kpts.shape[0] // shape[0]
    # query locations are at the coordinate frame of original image
    attn_map_pos = query_locations / down_rate

    # random pos
    x1 = img_vis_kpts.shape[1] * torch.rand(1)
    y1 = img_vis_kpts.shape[0] * torch.rand(1)
    x2 = img_vis_kpts.shape[1] * torch.rand(1)
    y2 = img_vis_kpts.shape[0] * torch.rand(1)
    random_pt_1 = [x1 / down_rate, y1 / down_rate]
    random_pt_2 = [x2 / down_rate, y2 / down_rate]
    attn_map_pos = attn_map_pos.tolist()
    attn_map_pos.append(random_pt_1)
    attn_map_pos.append(random_pt_2)

    fig, axs = plt.subplots(attn_layers_num, 20, figsize=(30, 8), )
    fig.subplots_adjust(
        bottom=0.07, right=0.97, top=0.98, left=0.03, wspace=0.00008, hspace=0.02,
    )

    for l in range(attn_layers_num):
        axs[l][0].imshow(img_vis_kpts)
        axs[l][0].set_ylabel("Enc.Att.\nLayer {}".format(l), fontsize=25)
        axs[l][0].set_xticks([])
        axs[l][0].set_yticks([])

    for id, attn_map in enumerate(enc_atten_maps_hwhw):
        for p_id, p in enumerate(attn_map_pos):
            if mode == "dependency":
                attention_map_for_this_point = F.interpolate(
                    attn_map[None, None, int(p[1]), int(p[0]), :, :],
                    scale_factor=down_rate,
                    mode="bilinear",
                )[0][0]
            else:
                attention_map_for_this_point = F.interpolate(
                    attn_map[None, None, :, :, int(p[1]), int(p[0])],
                    scale_factor=down_rate,
                    mode="bilinear",
                )[0][0]

            attention_map_for_this_point = (
                attention_map_for_this_point.squeeze().detach().cpu().numpy()
            )
            x, y = p[0] * down_rate, p[1] * down_rate
            img_vis_kpts_new = img_vis.copy()
            axs[id][p_id + 1].imshow(img_vis_kpts_new)
            if threshold is not None:
                mask = attention_map_for_this_point <= threshold
                attention_map_for_this_point[mask] = 0
                im = axs[id][p_id + 1].imshow(
                    attention_map_for_this_point, cmap="nipy_spectral", alpha=0.79
                )
            else:
                im = axs[id][p_id + 1].imshow(
                    attention_map_for_this_point, cmap="nipy_spectral", alpha=0.79
                )
            axs[id][p_id + 1].scatter(x=x, y=y, s=60, marker="*", c=kpt_color)
            axs[id][p_id + 1].set_xticks([])
            axs[id][p_id + 1].set_yticks([])
            if id == attn_layers_num - 1:
                axs[id][p_id + 1].set_xlabel(f"{p_id}", fontsize=25, )

    cax = plt.axes([0.975, 0.08, 0.005, 0.90])
    cb = fig.colorbar(
        im, cax=cax, ax=axs, orientation="vertical", fraction=0.05, aspect=50
    )
    cb.set_ticks([0.0, 0.5, 1])
    cb.ax.tick_params(labelsize=20)
    if save_img:
        plt.savefig("attention_map_{}_{}_{}.jpg".format(img_name, mode, model_name))
    plt.show()
