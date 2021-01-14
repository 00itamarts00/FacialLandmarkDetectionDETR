import json
import os
import pickle
from collections import namedtuple
from datetime import datetime

import cv2
import pandas as pd
import yaml


class FileHandler(object):
    def __init__(self):
        pass

    @staticmethod
    def load_yaml(yml_path):
        with open(yml_path) as file:
            dct_res = yaml.load(file, Loader=yaml.FullLoader)
        return dct_res

    @staticmethod
    def load_json(jsn_path):
        with open(jsn_path, 'r') as f:
            distros_dict = json.load(f)
        return distros_dict

    @staticmethod
    def dict2json(dictionary, fname):
        with open(fname, 'w') as fp:
            json.dump(dictionary, fp)

    @staticmethod
    def save_dict_to_pkl(self, dict_arg, dict_path):
        with open(dict_path, 'wb') as handle:
            pickle.dump(dict_arg, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pkl(pkl_path):
        with open(pkl_path, 'rb') as handle:
            b = pickle.load(handle)
        return b

    @staticmethod
    def load_csv(csv_filename):
        return pd.read_csv(csv_filename)

    @staticmethod
    def load_excel(xls_name):
        return pd.read_excel(xls_name)

    @staticmethod
    def make_folder_if_not_exist(fname):
        if not os.path.exists(fname):
            os.makedirs(fname)
        else:
            raise FileExistsError(f'folder {fname} already exists!')
        return fname

    @staticmethod
    def make_folder_if_not_exist_else_return_path(fname):
        if not os.path.exists(fname):
            os.makedirs(fname)
        else:
            return fname

    @staticmethod
    def check_if_folder_exists(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'directory {path} does not exist!')
        return path

    @staticmethod
    def save_image(img, fname):
        cv2.imwrite(fname, img)

    @staticmethod
    def dict_to_nested_namedtuple(d):
        MyTuple = namedtuple('MyTuple', d)
        my_tuple = MyTuple(**d)
        return my_tuple

    @staticmethod
    def get_datetime():
        dateTimeObj = datetime.now()
        return dateTimeObj.strftime("%d%m%y_%H%M%S")
