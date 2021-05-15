import math

import numpy as np
import skimage
import torch
from PIL import Image
from scipy.stats import multivariate_normal
from skimage import color

from common.resizer import fft_resize, fft_resize_to


def create_base_gaussian(dst_size, std_factor, magnify=20):
    xlim = (0, dst_size[1] - 1)
    ylim = (0, dst_size[0] - 1)
    xres = dst_size[1]
    yres = dst_size[0]

    x = np.linspace(xlim[0], xlim[1], xres)
    y = np.linspace(ylim[0], ylim[1], yres)
    xx, yy = np.meshgrid(x, y)
    xxyy = np.c_[xx.ravel(), yy.ravel()]

    # evaluate kernels at grid points
    std = np.eye(2) * std_factor

    mu_y = round(yres / 2)
    mu_x = round(xres / 2)
    k1 = multivariate_normal(mean=[mu_y, mu_x], cov=std)
    zz = k1.pdf(xxyy)
    img = zz.reshape((xres, yres)) * 100 * magnify
    img[img < 1e-12] = 0
    img = img.reshape((xres, yres))

    winsize = round(std_factor * 3)
    imga = np.copy(img[int(mu_y - winsize + 1):int(mu_y + winsize), int(mu_x - winsize + 1):int(mu_x + winsize)])
    return imga


def paste_slices(tup):
    pos, w, max_w = tup
    wall_min = max(pos, 0)
    wall_max = min(pos + w, max_w)
    block_min = -min(pos, 0)
    block_max = max_w - max(pos + w, max_w)
    block_max = block_max if block_max != 0 else None
    return slice(wall_min, wall_max), slice(block_min, block_max)


def paste(wall_, block_, loc):
    wall = np.copy(wall_)
    block = np.copy(block_)
    loc_zip = zip(loc, block.shape, wall.shape)
    wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
    wall[wall_slices] = block[block_slices]
    return wall


def create_heatmaps2(pts, im_size, dst_size, imga, res_factor):
    max_gv = np.max(imga)
    num_p = np.shape(pts)[0]
    pts_factor = im_size[0] / dst_size[0]
    hm_pts = np.copy(pts)
    hm_pts = hm_pts / pts_factor
    radga = round(imga.shape[0] / 2)
    heatmaps = np.zeros((num_p, dst_size[0], dst_size[1]))

    for idx_p in range(num_p):
        mu = (hm_pts[idx_p, :] * res_factor).round().astype(int)
        tmphm = np.zeros((dst_size[0] * res_factor, dst_size[1] * res_factor))
        try:
            tmphm = paste(tmphm, imga, (mu[1] - radga, mu[0] - radga))
        except:
            pass
        tmphm = tmphm[::res_factor, ::res_factor]
        heatmaps[idx_p] = np.copy(tmphm)

    heatmaps_ = np.divide(heatmaps, max_gv)
    return heatmaps_, hm_pts


def extract_pts_from_hm(hm, res_factor=5):
    num_p = hm.shape[0]
    pts = np.empty((num_p, 2))
    pts[:] = np.NaN
    for i in range(0, num_p):
        if res_factor > 1:
            hmr = fft_resize(hm[i, :, :], res_factor)
        else:
            hmr = hm[i, :, :]
        p = np.unravel_index(np.argmax(hmr), hmr.shape)
        pts[i, :] = (p[1], p[0])
    pts = np.true_divide(pts, res_factor)

    return pts


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


def decode_preds_heatmaps(output, res=[64, 64]):
    coords = get_preds(output)  # float type

    coords = coords.cpu()
    # post-processing
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

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds * (255 / res[0])


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

    #    imshowpts(im_, pts_)
    return im_, pts_


def gray2rgb_(image):
    try:
        h, w, c = image.shape
        return image
    except:
        h, w = image.shape
        return skimage.color.gray2rgb(image)


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def get_bbox(pts):
    minx = min(pts[:, 0])
    maxx = max(pts[:, 0])

    miny = min(pts[:, 1])
    maxy = max(pts[:, 1])
    return minx, maxx, miny, maxy


def get_cline(spts):
    lar = np.mean(spts[0:3], axis=0)
    rar = np.mean(spts[5:8], axis=0)

    upt = np.mean([lar, rar], axis=0)
    lpt = np.mean(spts[3:5], axis=0)

    cpt = np.mean([upt, lpt], axis=0)
    return upt, lpt, cpt


def image_scale(img, pts, dst_size):
    simg = fft_resize_to(img, dst_size)
    spts = np.copy(pts)
    scale0 = simg.shape[0] / img.shape[0]
    scale1 = simg.shape[1] / img.shape[1]
    spts[:, 0] = np.multiply(pts[:, 0], scale0)
    spts[:, 1] = np.multiply(pts[:, 1], scale1)
    return simg, spts


def image_rotate(simg, spts):
    upt, lpt, cpt = get_cline(spts)
    dx = upt[0] - lpt[0]
    dy = lpt[1] - upt[1]
    angle = np.arctan(dx / dy)
    rpts = rotate(spts, origin=cpt, degrees=-np.rad2deg(angle))
    sim_ = Image.fromarray(simg)
    rimg = np.array(sim_.rotate(np.rad2deg(angle)))

    return rimg, rpts


def image_pad(img, pts, per=50):
    sz0 = img.shape[0]
    sz1 = img.shape[1]

    padsz0 = int(round(per * sz0 / 100))
    padsz1 = int(round(per * sz1 / 100))

    padsz0s = int(round(padsz0 / 2))
    padsz0e = padsz0 - padsz0s
    padsz1s = int(round(padsz1 / 2))
    padsz1e = padsz1 - padsz1s

    pimg = np.pad(img, ((padsz0s, padsz0e), (padsz1s, padsz1e)), mode='constant', constant_values=(0, 0))
    ppts = np.copy(pts)
    ppts[:, 0] = pts[:, 0] + padsz1s  # Y,X are dims 0,1 at shape, but 1,0 at pts
    ppts[:, 1] = pts[:, 1] + padsz0s

    return pimg, ppts


def image_align(img, pts, dst_size=(1024, 1024)):
    pimg, ppts = image_pad(img, pts, per=50)

    rimg, rpts = image_rotate(pimg, ppts)
    timg, tpts = crop_image_by_pts(rimg, rpts, ppad=10)

    simg, spts = image_scale(timg, tpts, dst_size)

    return simg, spts
