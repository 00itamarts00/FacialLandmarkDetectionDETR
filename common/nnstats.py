import os

import numpy as np
import pandas as pd


class CnnStats(object):
    def __init__(self, stats_path, model):
        self.stats_path = stats_path
        self.v_mean_file = os.path.join(stats_path, 'nnstats_mean_val.csv')
        self.v_std_file = os.path.join(stats_path, 'nnstats_std_val.csv')
        self.g_mean_file = os.path.join(stats_path, 'nnstats_mean_grad.csv')
        self.g_std_file = os.path.join(stats_path, 'nnstats_std_grad.csv')

        os.makedirs(self.stats_path, exist_ok=True)

        self.df_v_mean = self.get_dataframe(self.v_mean_file, model)
        self.df_v_std = self.get_dataframe(self.v_std_file, model)
        self.df_g_mean = self.get_dataframe(self.g_mean_file, model)
        self.df_g_std = self.get_dataframe(self.g_std_file, model)

    def get_dataframe(self, stats_file, model):
        if os.path.isfile(stats_file):
            df = pd.read_csv(stats_file)
        else:
            df = pd.DataFrame(columns=('name', 'size'))

            for name, m in model.named_parameters():
                df.loc[len(df)] = {'name': name, 'size': np.array(m.size())}

            df.to_csv(stats_file)

        return df

    def add_measure(self, epoch, model):
        self.df_v_mean[f'{epoch}'] = None
        self.df_v_std[f'{epoch}'] = None
        self.df_g_mean[f'{epoch}'] = None
        self.df_g_std[f'{epoch}'] = None
        count = 0
        for name, m in model.named_parameters():
            self.df_v_mean[f'{epoch}'].loc[count] = m.data.mean().item()
            self.df_v_std[f'{epoch}'].loc[count] = m.data.std().item()
            if not m.grad == None:
                self.df_g_mean[f'{epoch}'].loc[count] = m.grad.mean().item()
                self.df_g_std[f'{epoch}'].loc[count] = m.grad.std().item()

            count = count + 1

    def dump(self):
        self.df_v_mean.to_csv(self.v_mean_file)
        self.df_v_std.to_csv(self.v_std_file)
        self.df_g_mean.to_csv(self.g_mean_file)
        self.df_g_std.to_csv(self.g_std_file)
