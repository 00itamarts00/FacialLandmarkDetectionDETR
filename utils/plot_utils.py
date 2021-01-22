import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')


def plot_ldm_on_image(image, pts, channels_last=False):
    plt.figure()
    pts = np.array(pts)
    if not channels_last:
        plt.imshow(np.swapaxes(np.swapaxes(image, 0, -1), 0, 1))
        plt.scatter(pts.T[0], pts.T[1], s=2)
