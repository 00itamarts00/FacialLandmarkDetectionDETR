import os

from create_dataset_utils.consts import DATASET_PATH, OUTPUT_PATH
from create_dataset_utils.dataset_utils import all_pts_files_in_dataset, read_pair_img_pts, crop_image_koi, save_pair_img_pts


class CreateDataset(object):
    def __init__(self):
        self.pts_paths = all_pts_files_in_dataset(DATASET_PATH)

    def run(self):
        for basename in self.pts_paths:
            # if basename != '300W/02_Outdoor/outdoor_177':
            #     continue
            print(basename)
            image, kps_oi = read_pair_img_pts(os.path.join(DATASET_PATH, basename))
            img_cropped, kps_io_cropped = crop_image_koi(image, kps_oi, margin=0.1, border='linear_ramp')
            save_pair_img_pts(destination=OUTPUT_PATH, basename=basename, img_cropped=img_cropped,
                              kps_io_cropped=kps_io_cropped, debug=True)


def main():
    dataset = CreateDataset()
    dataset.run()


if __name__ == "__main__":
    main()
