import os

import numpy as np
import pandas as pd
from tqdm import tqdm


class CnnStats(object):
    def __init__(self, stats_path, model):
        self.stats_path = stats_path
        self.v_mean_file = os.path.join(stats_path, "nnstats_mean_val.csv")
        self.v_std_file = os.path.join(stats_path, "nnstats_std_val.csv")
        self.g_mean_file = os.path.join(stats_path, "nnstats_mean_grad.csv")
        self.g_std_file = os.path.join(stats_path, "nnstats_std_grad.csv")

        self.df_v_mean = self.get_dataframe(self.v_mean_file, model)
        self.df_v_std = self.get_dataframe(self.v_std_file, model)
        self.df_g_mean = self.get_dataframe(self.g_mean_file, model)
        self.df_g_std = self.get_dataframe(self.g_std_file, model)

    @staticmethod
    def get_dataframe(stats_file, model):
        df = (
            pd.read_csv(stats_file)
            if os.path.isfile(stats_file)
            else pd.DataFrame(columns=("name", "size"))
        )

        for name, m in tqdm(model.named_parameters()):
            df.loc[len(df)] = {"name": name, "size": np.array(m.size())}

        df.to_csv(stats_file)
        return df

    def add_measure(self, epoch, model, dump=False):
        self.df_v_mean[f"{epoch}"] = None
        self.df_v_std[f"{epoch}"] = None
        self.df_g_mean[f"{epoch}"] = None
        self.df_g_std[f"{epoch}"] = None
        count = 0
        for name, m in model.named_parameters():
            self.df_v_mean[f"{epoch}"].loc[count] = m.data.mean().item()
            self.df_v_std[f"{epoch}"].loc[count] = m.data.std().item()
            if not m.grad == None:
                self.df_g_mean[f"{epoch}"].loc[count] = m.grad.mean().item()
                self.df_g_std[f"{epoch}"].loc[count] = m.grad.std().item()
            count = count + 1

        if dump:
            self.dump()

    def dump(self):
        self.df_v_mean.to_csv(self.v_mean_file)
        self.df_v_std.to_csv(self.v_std_file)
        self.df_g_mean.to_csv(self.g_mean_file)
        self.df_g_std.to_csv(self.g_std_file)
