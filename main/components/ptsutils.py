import copy
import math

import numpy as np
import skimage
import torch
from skimage import color

MATCHED_PARTS = {
    "300W": ([1, 17], [2, 16], [3, 15], [4, 14], [5, 13], [6, 12], [7, 11], [8, 10],
             [18, 27], [19, 26], [20, 25], [21, 24], [22, 23],
             [32, 36], [33, 35],
             [37, 46], [38, 45], [39, 44], [40, 43], [41, 48], [42, 47],
             [49, 55], [50, 54], [51, 53], [62, 64], [61, 65], [68, 66], [59, 57], [60, 56]),
    "MENPO": ([1, 6], [2, 5], [3, 4],
              [7, 12], [8, 11], [9, 10],
              [13, 15], [16, 18]),
    "COFW": ([1, 2], [5, 7], [3, 4], [6, 8], [9, 10], [11, 12], [13, 15], [17, 18], [14, 16], [19, 20], [23, 24]),
    "WFLW": ([0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23], [10, 22],
             [11, 21], [12, 20], [13, 19], [14, 18], [15, 17],  # check
             [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50], [39, 49], [40, 48], [41, 47],  # elbrow
             [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75], [66, 74], [67, 73],
             [55, 59], [56, 58],
             [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],
             [88, 92], [89, 91], [95, 93], [96, 97])}


def get_face68_flip():
    def vx(st, en=None, step=1):
        if en == None:
            return np.array(range(st, st + 1))

        exen = 1 if step > 0 else -1
        return np.array(range(st, en + exen, step))

    dl = list()
    dl.append([vx(1, 17), vx(17, 1, -1)])
    dl.append([vx(18, 22), vx(27, 23, -1)])
    dl.append([vx(23, 27), vx(22, 18, -1)]),
    dl.append([vx(28, 31), vx(28, 31)]),
    dl.append([vx(32, 36), vx(36, 32, -1)]),
    dl.append([vx(37, 40), vx(46, 43, -1)]),
    dl.append([vx(41), vx(48)]),
    dl.append([vx(42), vx(47)]),
    dl.append([vx(43, 46), vx(40, 37, -1)]),
    dl.append([vx(47), vx(42)]),
    dl.append([vx(48), vx(41)]),
    dl.append([vx(49, 55), vx(55, 49, -1)]),
    dl.append([vx(56, 60), vx(60, 56, -1)]),
    dl.append([vx(61, 65), vx(65, 61, -1)]),
    dl.append([vx(66, 68), vx(68, 66, -1)])

    sidx, didx = list(), list()
    for i in range(len(dl)):
        didx = didx + np.array(dl[i][0]).tolist()
        sidx = sidx + np.array(dl[i][1]).tolist()

    return np.asarray(sidx) - 1, np.asarray(didx) - 1


def fliplr_img_pts(im, pts, width=None):
    width = im.shape[0] if width is None else width
    ima = fliplr_img(im)
    ptsa = fliplr_joints(pts, width=width)
    return ima, ptsa


def fliplr_img(im):
    return np.fliplr(im)


def fliplr_ldmk(pts, img_width):
    ptsa = copy.deepcopy(pts)

    sidx, didx = get_face68_flip()

    ptsa[didx, 0] = pts[sidx, 0]
    ptsa[didx, 1] = pts[sidx, 1]
    ptsa[didx, 0] = img_width - ptsa[didx, 0]
    return ptsa


def fliplr_img_pts_ver2(img, pts, dataset):
    img_ = np.fliplr(img)
    parent_dataset = '300W' if dataset in ['HELEN', 'LFPW', 'IBUG'] else dataset
    pts_ = fliplr_joints(pts, img_.shape[0], dataset=parent_dataset)
    return img_, pts_


def fliplr_joints(x, width, dataset='300W'):
    """
    flip coords
    """
    np_detached = lambda t: t.cpu().detach().numpy() if not isinstance(t, np.ndarray) else t
    np.array([np_detached(i) for i in x])

    matched_parts = MATCHED_PARTS[dataset]
    # Flip horizontal
    x[:, 0] = width - x[:, 0]

    if dataset == 'WFLW':
        for pair in matched_parts:
            tmp = x[pair[0], :].copy()
            x[pair[0], :] = x[pair[1], :]
            x[pair[1], :] = tmp
    else:
        for pair in matched_parts:
            tmp = np_detached(x[pair[0] - 1, :]).copy()
            x[pair[0] - 1, :] = np_detached(x[pair[1] - 1, :])
            x[pair[1] - 1, :] = tmp
    return x


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


def decode_preds_heatmaps(output, hm_size=None):
    if hm_size is None:
        hm_size = [64, 64]
    coords = get_preds(output)  # float type

    coords = coords.cpu()
    # post-processing
    for n, (img, coord) in enumerate(zip(output, coords)):
        for p, hm in enumerate(img):
            px, py = coord[p][0].floor().int().item(), coord[p][1].floor().int().item()
            if (px > 1) and (px < hm_size[0]) and (py > 1) and (py < hm_size[1]):
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1] - hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds * (256 / hm_size[0])


def crop_image_by_pts(im, pts, ppad=20):
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    fw = max_x - min_x
    fh = max_y - min_y

    wpad = fw * ppad / 100
    hpad = fh * ppad / 100

    if len(np.shape(im)) > 2:
        [imh, imw, ch] = np.shape(im)
    else:
        [imh, imw] = np.shape(im)
        ch = 1

    min_xp = int(round(np.max([min_x - wpad, 0])))
    max_xp = int(round(np.min([max_x + wpad, imw])))

    min_yp = int(round(np.max([min_y - hpad, 0])))
    max_yp = int(round(np.min([max_y + hpad, imh])))

    nw = max_xp - min_xp
    nh = max_yp - min_yp

    if nw > nh:
        hnpad = (nw - nh) / 2
        min_yp = int(round(np.max([min_yp - hnpad, 0])))
        max_yp = int(round(np.min([max_yp + hnpad, imh])))
    else:
        wnpad = (nh - nw) / 2
        min_xp = int(round(np.max([min_xp - wnpad, 0])))
        max_xp = int(round(np.min([max_xp + wnpad, imw])))

    nw = max_xp - min_xp
    nh = max_yp - min_yp

    nw = math.floor(min(nw, imw) / 2) * 2
    nh = math.floor(min(nh, imh) / 2) * 2

    imc = np.copy(im[min_yp:min_yp + nh, min_xp:min_xp + nw])

    pts_ = np.copy(pts)
    pts_[:, 0] = pts_[:, 0] - min_xp
    pts_[:, 1] = pts_[:, 1] - min_yp

    dim = max(nh, nw)

    if ch == 1:
        im_ = np.full((dim, dim), 0, dtype=np.uint8)
    else:
        im_ = np.full((dim, dim, ch), 0, dtype=np.uint8)

    xx = (dim - nw) // 2
    yy = (dim - nh) // 2

    # copy img image into center of result image
    im_[yy:yy + nh, xx:xx + nw] = imc

    pts_[:, 0] = pts_[:, 0] + xx
    pts_[:, 1] = pts_[:, 1] + yy
    return im_, pts_


def gray2rgb_(image):
    try:
        h, w, c = image.shape
        return image
    except:
        h, w = image.shape
        return skimage.color.gray2rgb(image)


def get_bbox(pts):
    minx = min(pts[:, 0])
    maxx = max(pts[:, 0])

    miny = min(pts[:, 1])
    maxy = max(pts[:, 1])
    return minx, maxx, miny, maxy


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals
