import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')


def plot_ldm_on_image(image, pts, channels_last=False):
    plt.figure()
    pts = np.array(pts)
    if not channels_last:
        plt.imshow(image)
        plt.scatter(pts.T[0], pts.T[1], s=2)
