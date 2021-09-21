import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import ImageGrid

warnings.filterwarnings('ignore')


def closest_divisors(n):
    a = round(math.sqrt(n))
    while n % a > 0:
        a -= 1
    return a, n // a


def plot_score_maps(item, index, score_map, predictions):
    dataset = item['dataset'][index]
    img_name = item['img_name'][index]
    img = item['img'].numpy()[index]
    score_map_i = score_map.numpy()[index]
    preds = predictions.numpy()[index]
    opts = item['opts'].numpy()[index]
    cnt = 1
    fig = Figure(figsize=[30, 20])
    canvas = FigureCanvas(fig)
    fig.suptitle(f'debug_image\ndataset: {dataset} name: {img_name}')
    ax = fig.add_subplot(8, 9, cnt)
    ax.imshow(np.swapaxes(np.swapaxes(img, 0, -1), 0, 1))
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    cnt += 1
    ax = fig.add_subplot(8, 9, cnt)
    ax.scatter(opts.T[0], opts.T[1], s=1, c='b', label='gt')
    ax.scatter(preds.T[0], preds.T[1], s=1, c='r', label='pred')
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_title(f'pred and target', fontsize=5)
    cnt += 1
    for hm in score_map_i:
        ax = fig.add_subplot(8, 9, cnt)
        ax.imshow(hm, cmap='gray')
        ax.set_title(cnt, fontsize=5)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        cnt += 1
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return image


def scatter_prediction_gt(pred, gt):
    plt.figure()
    plt.scatter(pred.T[1], pred.T[0], s=2, label='pred')
    plt.scatter(gt.T[1], gt.T[0], s=2, label='gt')
    plt.legend()
    plt.title('scatter pred and gt')
    plt.show()
    plt.pause(0.01)


def plot_gt_pred_on_img(item, predictions, index):
    dataset = item['dataset'][index]
    img_name = item['img_name'][index]
    img = item['img'].numpy()[index]
    img = renorm_image(img)
    img = np.array(img).astype(np.uint8)
    fig = Figure()
    canvas = FigureCanvas(fig)
    preds = predictions[index].detach().cpu() if not isinstance(predictions, np.ndarray) else predictions[index]
    tpts = item['tpts'].numpy()[index]
    ax = fig.add_subplot(111)
    ax.set_title(f'debug_image\ndataset: {dataset} name: {img_name}')
    ax.imshow(img)
    ax.scatter(preds[:, 0], preds[:, 1], s=5, c='r', label='pred')
    ax.scatter(tpts.T[0], tpts.T[1], s=5, c='b', label='gt')
    for ptp, ptgt in zip(preds, tpts):
        ax.plot([ptp[0], ptgt[0]], [ptp[1], ptgt[1]], 'g-', linewidth=0.5)
    ax.legend()
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return image


def renorm_image(img):
    mean = np.array([[0.5021, 0.3964, 0.3471]], dtype=np.float32)
    std = np.array([0.2858, 0.2547, 0.2488], dtype=np.float32)
    img_ = np.array(img).transpose([1, 2, 0])
    img_ = 256 * (img_ * std + mean)
    img_ = np.clip(img_, a_min=0, a_max=255)
    return np.ubyte(img_)


def plot_grid_of_ldm(dataset, imgs, preds, tpts):
    fig = plt.figure(figsize=(10., 6.))
    canvas = FigureCanvas(fig)
    w, h = closest_divisors(imgs.__len__())
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(w, h),  # creates 16x16 grid of axes
                     axes_pad=0.0)  # pad between axes in inch.
    for ax, img, pred, tpt in zip(grid, imgs, preds, tpts):
        img = renorm_image(img)
        img = np.array(img).astype(np.uint8)
        ax.imshow(img)
        ax.axis('off')

        a = ax.scatter(pred.T[0], pred.T[1], s=1, c='r', label='pred')
        b = ax.scatter(tpt.T[0], tpt.T[1], s=1, c='b', label='gt')
        for ptp, ptgt in zip(pred, tpt):
            ax.plot([ptp[0], ptgt[0]], [ptp[1], ptgt[1]], 'g-', linewidth=0.5)

    plt.suptitle(f'Dataset: {dataset}\nToughest Predictions', y=0.98)
    fig.legend((a, b), ['pred', 'gt'], loc='lower center')

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return image
