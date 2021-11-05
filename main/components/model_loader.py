import logging
import os

import torch

from common.s3_interface import download_file_from_s3
from main.detr.models.detr import build as build_model
from main.globals import *
from models.HRNET.HRNET import get_face_alignment_net
from models.PERCIEVER import Perceiver
from models.TRANSPOSE.transpose_h import get_pose_net as get_pose_hrnet
from models.TRANSPOSE.transpose_r import get_pose_net as get_pose_resnet


def load_model(model_name, params, pretrained_path=None):
    use_pretrained = True if pretrained_path is not None else False
    model = None
    if model_name == PERC:
        model_args = params.perciever_args
        model = Perceiver(args=model_args)
        if use_pretrained:
            load_pretrained_model(model, pretrained_path)

    if model_name == DETR:
        model_args = params.detr_args
        model = build_model(args=model_args)
        if use_pretrained:
            if not os.path.exists(pretrained_path):
                os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
                download_file_from_s3(s3_object_key=os.path.basename(pretrained_path), local_file_name=pretrained_path)
            load_pretrained_model(model, pretrained_path)

    if model_name == TRANSPOSE:
        model_args = params.transpose_args
        get_pose_net = get_pose_hrnet if model_args.backbone == HRNET.lower() else get_pose_resnet
        model = get_pose_net(cfg=model_args)
        if use_pretrained:
            if not os.path.exists(pretrained_path):
                os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
                download_file_from_s3(s3_object_key=os.path.basename(pretrained_path),
                                      local_file_name=pretrained_path)
            load_pretrained_model(model, pretrained_path)

    if model_name == HRNET:
        model_args = params.hrnet_args
        model = get_face_alignment_net(config=model_args)

        if use_pretrained:
            if not os.path.exists(pretrained_path):
                os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
                download_file_from_s3(s3_object_key=os.path.basename(pretrained_path),
                                      local_file_name=pretrained_path)
            load_pretrained_model(model, pretrained_path)

    return model


def load_pretrained_model(model, pretrained_path, strict=True):
    logging.info(msg=f'Loading pretrained model: {pretrained_path}')
    model_best_state = torch.load(pretrained_path)
    model.load_state_dict(model_best_state['state_dict'], strict=strict)
