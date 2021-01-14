# Original model from paper
# Same as HMLM01A - but added 3 additional branches. Initilized by copy_weights script from HMLM01A(ep100)
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src_code.common.ptsutils import extract_pts_from_hm


class FT(nn.Module):
    def __init__(self, num_channels=1, bnorm=False):
        super(FT, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 68, 3, 1, padding=1),
            nn.BatchNorm2d(68),
            nn.ReLU(inplace=False),
            # nn.Dropout2d(0.5),
        )

    def forward(self, x):
        x = self.features(x)
        return x


class HM(nn.Module):
    def __init__(self, input_depth=68, npts=68):
        super(HM, self).__init__()
        self.hm_regressor = nn.Sequential(
            nn.Conv2d(input_depth, 64, 7, 1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 13, 1, padding=6),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            # nn.Dropout2d(0.5),
            nn.Conv2d(128, npts, 1, 1),
            nn.BatchNorm2d(npts),
            # nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x = self.hm_regressor(x)
        return x


class Net(nn.Module):
    def __init__(self, npts=68, num_channels=3):
        super(Net, self).__init__()

        self.num_branches_aloc = 8
        self.num_branches = 4
        self.npts = npts
        self.ft0 = FT(num_channels=num_channels)
        self.hm0 = HM(input_depth=npts)

        self.ft1 = FT(num_channels=num_channels)

        self.hm1 = HM(input_depth=npts)
        self.hm2 = HM(input_depth=2 * npts)
        self.hm3 = HM(input_depth=2 * npts)
        self.hm4 = HM(input_depth=2 * npts)

        # not in use for meanwhile. Attend to use ist in later versions.
        self.hm5 = HM(input_depth=2 * npts)
        self.hm6 = HM(input_depth=2 * npts)
        self.hm7 = HM(input_depth=2 * npts)
        self.hm8 = HM(input_depth=2 * npts)

        self.output_branch = 'HM03'
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, x):
        xft1 = self.ft1(x)
        xhm = [None] * self.num_branches
        xhm0 = self.hm1(xft1)
        xhm1 = self.hm2(torch.cat((xft1, xhm0), 1))
        xhm2 = self.hm3(torch.cat((xft1, xhm1), 1))
        xhm3 = self.hm4(torch.cat((xft1, xhm2), 1))
        return [xhm0, xhm1, xhm2, xhm3]

    def loss(self, output, target):
        loss0 = self.loss_fn(output[0], target)
        loss1 = self.loss_fn(output[1], target)
        loss2 = self.loss_fn(output[2], target)
        loss3 = self.loss_fn(output[3], target)
        loss = loss0 + loss1 + loss2 + loss3
        return loss

    def extract_epts(self, output, hm_factor=4, res_factor=5):
        num_i = len(output[0])   # num samples
        num_b = len(output)  # num branches
        col = []
        for idx_b in range(num_b):
            col.append(f'HM{idx_b:02}')
        col.append(f'output')
        row = np.array(range(num_i))
        df = pd.DataFrame(columns=col, index=row)

        for idx_i in range(num_i):
            for idx_b in range(num_b):
                ehms = output[idx_b].cpu().detach().numpy()
                epts = extract_pts_from_hm(ehms[idx_i], res_factor=res_factor)
                epts_ = np.multiply(epts, hm_factor)
                df[f'HM{idx_b:02}'].loc[idx_i] = epts_
            df[f'output'] = df[self.output_branch]
        return df

    def init_weights(self, pretrained=''):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            print('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                print('=> loading {} pretrained model {}'.format(k, pretrained))

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


def get_instance(pretrained=''):
    model = Net(npts=68)
    model.init_weights(pretrained)

    return model
