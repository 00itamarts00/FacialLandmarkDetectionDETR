import matplotlib.pyplot as plt
import menpo.io as mio
import numpy as np


class LMDK_Images(object):
    def __init__(self, filename):
        self.filename = filename
        self.load()

    def load(self):
        self.img = mio.import_image(self.filename)

    @property
    def points(self):
        return np.array(self.img.landmarks['PTS'].points, dtype=float)

    @property
    def num_landmarks(self):
        return self.img.landmarks['PTS'].n_points

    def plot_lmdk_img(self):
        self.img.view_landmarks(render_numbering=True, render_lines=True, marker_style='.')
        plt.show()

    def get_cropped_image(self, boundary=30):
        self.img.crop_to_landmarks(boundary=boundary)

    def get_cropped_landmarks(self):
        np.array(self.get_cropped_image().landmarks['PTS'].points, dtype=float)


def main():
    filename = '/home/itamar/thesis/DATASET/WS03/300W/02_Outdoor/outdoor_005.png'
    aaa = LMDK_Images(filename=filename)


if __name__ == "__main__":
    main()
