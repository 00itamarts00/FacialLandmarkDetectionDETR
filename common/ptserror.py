from __future__ import print_function

import numpy as np


def distance(v1, v2):
    if len(v1.shape) > 1:
        d = np.sqrt(np.sum(np.power(np.subtract(v1, v2), 2), 1))
    else:
        d = np.sqrt(np.sum(np.power(np.subtract(v1, v2), 2)))
    return d


def calc_pts_error(epts, opts, normp=(36, 45)):  # for 68 points
    r = distance(opts[normp[0]], opts[normp[1]])
    d = distance(epts, opts)
    err = np.divide(d, r)

    return np.mean(err), err


class ptserror:
    def __init__(self):
        self.mean_err = []
        self.std_err = []
        self.max_err = []

        self.time = dict()

        self.atime_pre = []
        self.atime_cnn = []

    def add_results(self, epts, opts, normp=(36, 45)):
        r = distance(opts[normp[0]], opts[normp[1]])
        d = distance(epts, opts)
        err = np.divide(d, r)
        self.mean_err.append(np.mean(err))
        self.max_err.append(np.max(err))
        self.std_err.append(np.std(err))

    def add_time(self, atime_pre, atime_cnn):
        self.atime_pre.append(atime_pre)
        self.atime_cnn.append(atime_cnn)

    def get_error(self):
        mean_err_ = np.array(self.mean_err).flatten()
        max_err_ = np.array(self.max_err).flatten()
        std_err_ = np.array(self.std_err).flatten()

        errvals = dict()
        errvals['max'] = np.max(max_err_)
        errvals['mean'] = np.mean((mean_err_))
        errvals['std'] = np.std((std_err_))
        errvals['max_mean'] = np.max((mean_err_))

        errvals['ecntmax_003'] = np.sum(max_err_ > 0.03) / len(max_err_)
        errvals['ecntmax_005'] = np.sum(max_err_ > 0.05) / len(max_err_)
        errvals['ecntmax_010'] = np.sum(max_err_ > 0.10) / len(max_err_)

        errvals['ecntmean_003'] = np.sum(mean_err_ > 0.03) / len(max_err_)
        errvals['ecntmean_005'] = np.sum(mean_err_ > 0.05) / len(max_err_)
        errvals['ecntmean_010'] = np.sum(mean_err_ > 0.10) / len(max_err_)

        errvals['atime_pre'] = np.mean((self.atime_pre))
        errvals['atime_cnn'] = np.mean((self.atime_cnn))

        return errvals

    def estring(self, errvals):
        str = '\t max: {:.3f} \t mean: {:.3f}  \t max_mean: {:.3f}  \t ecntmax_005: {:.3f} \t ecntmean_005: {:.3f}'.format(
            errvals['max'], errvals['mean'], errvals['max_mean'], errvals['ecntmax_005'], errvals['ecntmean_005'])
        return str
