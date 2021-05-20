import glob
import os

import pandas as pd

from utils.file_handler import FileHandler


class DataOrganize(object):
    def __init__(self, pointers, params):
        self.pointers = pointers
        self.params = params

    @property
    def ds_params(self):
        return self.params['dataset']

    @property
    def dataset_dir(self):
        return self.pointers['dataset_dir']

    @staticmethod
    def get_base_names(path):
        ls = glob.glob(os.path.join(path, '*'))
        basenames = [os.path.splitext(os.path.relpath(i, path))[0] for i in ls]
        return basenames

    def get_single_dataset_metadata(self, dataset):
        ds_path = os.path.join(self.dataset_dir, dataset)
        ls = os.listdir(ds_path)
        if dataset == 'helen':
            print('here')

        if dataset == 'COFW68':
            ds_path = os.path.join(self.dataset_dir, dataset, 'COFW_test_color')
            ls = os.listdir(ds_path)
        if 'img' in ls:
            if dataset == '300W' or dataset == 'WFLW/trainset' or dataset == 'WFLW/testset':
                basenames = self.get_base_names(os.path.join(ds_path, 'img', '*'))
                basenames = [i[3:] for i in basenames]
            else:
                basenames = self.get_base_names(os.path.join(ds_path, 'img'))
            datasets_ser = [dataset for i in basenames]
            df = pd.DataFrame(list(zip(basenames, datasets_ser)), columns=['basename', 'dataset'])
            return df

    def unify_metadata(self, type='train'):
        datasets = self.params['train']['datasets']
        df = pd.DataFrame(columns=['basename', 'dataset'])
        for dataset in datasets:
            dataset_csv = dataset.replace('/', '_') + '.csv'
            dsdf = pd.read_csv(os.path.join(self.pointers['dataset_dir'], dataset_csv))
            df = pd.concat([df, dsdf], ignore_index=True)
        return df

    def run(self):
        datasets = self.ds_params['to_use']
        for dataset in datasets:
            print(dataset)
            dataset_csv = dataset.replace('/', '_') + '.csv'
            out_path = os.path.join(self.dataset_dir, dataset_csv)
            if not os.path.exists(out_path):
                df = self.get_single_dataset_metadata(dataset=dataset)
                out_path.replace('/', '_')
                df.to_csv(out_path)
        df = self.unify_metadata()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    params = FileHandler.load_yaml('../main/params.yaml')
    pointers = FileHandler.load_yaml('pointers.yaml')

    dsorg = DataOrganize(pointers=pointers, params=params)
    dsorg.run()


if __name__ == '__main__':
    main()
