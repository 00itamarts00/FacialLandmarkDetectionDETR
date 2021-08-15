import os
from collections import OrderedDict
from pathlib import Path
from shutil import copyfile

import numpy as np
from tqdm import tqdm

from utils.file_handler import FileHandler


def convert_wflw98_to_wflw68():
    DLIB_68_TO_WFLW_98_IDX_MAPPING = OrderedDict()
    DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(0, 17), range(0, 34, 2))))  # jaw | 17 pts
    DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(17, 22), range(33, 38))))  # left upper eyebrow points | 5 pts
    DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(22, 27), range(42, 47))))  # right upper eyebrow points | 5 pts
    DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(27, 36), range(51, 60))))  # nose points | 9 pts
    DLIB_68_TO_WFLW_98_IDX_MAPPING.update({36: 60})  # left eye points | 6 pts
    DLIB_68_TO_WFLW_98_IDX_MAPPING.update({37: 61})
    DLIB_68_TO_WFLW_98_IDX_MAPPING.update({38: 63})
    DLIB_68_TO_WFLW_98_IDX_MAPPING.update({39: 64})
    DLIB_68_TO_WFLW_98_IDX_MAPPING.update({40: 65})
    DLIB_68_TO_WFLW_98_IDX_MAPPING.update({41: 67})
    DLIB_68_TO_WFLW_98_IDX_MAPPING.update({42: 68})  # right eye | 6 pts
    DLIB_68_TO_WFLW_98_IDX_MAPPING.update({43: 69})
    DLIB_68_TO_WFLW_98_IDX_MAPPING.update({44: 71})
    DLIB_68_TO_WFLW_98_IDX_MAPPING.update({45: 72})
    DLIB_68_TO_WFLW_98_IDX_MAPPING.update({46: 73})
    DLIB_68_TO_WFLW_98_IDX_MAPPING.update({47: 75})
    DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(48, 68), range(76, 96))))  # mouth points | 20 pts

    WFLW_98_TO_DLIB_68_IDX_MAPPING = {v: k for k, v in DLIB_68_TO_WFLW_98_IDX_MAPPING.items()}
    return WFLW_98_TO_DLIB_68_IDX_MAPPING


def get_pts_from_txt(fname):
    pts_str = FileHandler.read_general_txt_file(fname).split('\n')
    pts_start = pts_str.index('{') + 1
    pts_end = pts_str.index('}')
    pts_txt = pts_str[pts_start:pts_end]
    pts = np.array([np.array(i.split()).astype(float) for i in pts_txt])
    pts = [tuple(i) for i in pts]
    return pts


def create_pts_files_from_full_annotation_file(ann_file, folder):
    f = open(ann_file, "r")
    lines = f.readlines()
    for line in tqdm(lines):
        name, pts = line.split()[0], np.array(line.split()[1:]).astype(float)
        pts = pts.reshape(len(pts) // 2, 2)
        name = os.path.basename(name.replace('jpg', 'pts'))
        with open(os.path.join(folder, name), 'w') as ann:
            ann.write('version: 1\n')
            ann.write(f'n_points: {len(pts)}\n')
            ann.write('{\n')
            for pt in pts:
                ann.write(' '.join(pt.astype('str')) + '\n')
            ann.write('}')
        ann.close()
    f.close()
    print('Done')


def create_img_pts_files_from_full_annotation_file_wflw(ann_file, folder, type='trainset', num_landmarks=68):
    f = open(ann_file, "r")
    lines = f.readlines()
    for line in tqdm(lines):
        pts = np.array(line.split()[:196]).astype(float)
        pts = pts.reshape(len(pts) // 2, 2)
        if num_landmarks == 68:
            WFLW_98_TO_DLIB_68_IDX_MAPPING = convert_wflw98_to_wflw68()
            pts = np.array([pts[k] for k, v in WFLW_98_TO_DLIB_68_IDX_MAPPING.items()])
        name = line.split()[-1]
        name_pts = name.replace('jpg', 'pts')
        dbb = np.array(line.split()[196:200]).astype(int)
        pose, expression, illumination, make_up, occlusion, blur = np.array(line.split()[200:206]).astype(bool)
        if not os.path.exists(os.path.dirname(Path(folder).parent / type / name)):
            os.makedirs(os.path.dirname(Path(folder).parent / type / name))
        copyfile(Path(folder) / name, os.path.join(Path(folder).parent / type / name))
        with open(Path(folder).parent / type / name_pts, 'w') as ann:
            ann.write('version: 1\n')
            ann.write(f'n_points: {len(pts)}\n')
            ann.write('{\n')
            for pt in pts:
                ann.write(' '.join(pt.astype('str')) + '\n')
            ann.write('}\n')
            # ann.write(f'pose: {pose}\n')
            # ann.write(f'expression: {expression}\n')
            # ann.write(f'make_up: {make_up}\n')
            # ann.write(f'illumination: {illumination}\n')
            # ann.write(f'occlusion: {occlusion}\n')
            # ann.write(f'blur: {blur}\n')
            # ann.write(f'dbb: {dbb}\n')
        ann.close()
    f.close()
    print('Done')


def create_pts_img_for_cofw_color(mat_file):
    raise NotImplementedError


def main():
    ann_file = '/home/itamar/thesis/DATASET/WS03/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt'
    folder = '/home/itamar/thesis/DATASET/WS03/WFLW/WFLW_images'
    mat_file = '/home/itamar/thesis/DATASET/WS03/COFW_color/COFW_train_color.mat'
    create_pts_img_for_cofw_color(mat_file)
    create_img_pts_files_from_full_annotation_file_wflw(ann_file=ann_file, folder=folder, type='trainset')


if __name__ == "__main__":
    main()
