import numpy as np

from common.resizer import fft_resize


def create_heatmaps2(pts, im_size, dst_size, sigma_gauss):
    # if type == "multiple" :         num_p =1
    num_p = len(pts)
    pts_factor = im_size[0] / dst_size[0]
    hm_pts = np.copy(pts) / pts_factor
    heatmaps = np.zeros((num_p, dst_size[0], dst_size[1]))
    mu_vec = np.floor(hm_pts).astype(int)
    sigma_vec = np.ones_like(mu_vec) * sigma_gauss
    for i, (mu, sigma) in enumerate(zip(mu_vec, sigma_vec)):
        heatmaps[i] = np.clip(make_gaussian2d(mu, sigma, dst_size, theta=0), a_min=1e-3, a_max=1.5)
    return heatmaps, hm_pts


def make_gaussian2d(mu, sigma, out_size, theta=0):
    # x_center and y_center will be the center of the gaussian, theta will be the rotation angle
    # sigma_x and sigma_y will be the stdevs in the x and y axis before rotation
    # x_size and y_size give the size of the frame
    sx, sy = sigma
    x0, y0 = mu
    theta = 2 * np.pi * theta / 360
    x_size, y_size = out_size
    x = np.arange(0, x_size, 1, float)
    y = np.arange(0, y_size, 1, float)
    y = y[:, np.newaxis]

    # rotation
    a = np.cos(theta) * x - np.sin(theta) * y
    b = np.sin(theta) * x + np.cos(theta) * y
    a0 = np.cos(theta) * x0 - np.sin(theta) * y0
    b0 = np.sin(theta) * x0 + np.cos(theta) * y0

    return np.exp(-(((a - a0) ** 2) / (2 * (sx ** 2)) + ((b - b0) ** 2) / (2 * (sy ** 2))))


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
