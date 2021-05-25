from __future__ import print_function

import numpy as np
import scipy.fftpack as sfft  # you have to import this package


# D = abs(ifft2(fftshift(padarray(fftshift(fft2(ra)),size(ra),0,'both'))));


def fft_resize(Z, sfactor):
    sz0 = Z.shape[0]
    sz1 = Z.shape[1]
    dsz0 = sfactor * sz0
    dsz1 = sfactor * sz1
    padsz0 = int(round((dsz0 - sz0) / 2))
    padsz1 = int(round((dsz1 - sz1) / 2))

    Zf = sfft.fftshift(sfft.fft2(Z))

    Zf = np.pad(
        Zf,
        ((padsz0, padsz0), (padsz1, padsz1)),
        mode="constant",
        constant_values=(0, 0),
    )
    Zr = sfft.ifft2(sfft.fftshift(Zf))
    return np.abs(Zr)


def fft_resize_to(Z, dstsz):
    padsz0 = dstsz[0] - Z.shape[0]
    padsz1 = dstsz[1] - Z.shape[1]

    padsz0s = int(round(padsz0 / 2))
    padsz0e = padsz0 - padsz0s
    padsz1s = int(round(padsz1 / 2))
    padsz1e = padsz1 - padsz1s

    Zf = sfft.fftshift(sfft.fft2(Z))

    if padsz0 < 0:
        Zf = Zf[-padsz0s:padsz0e, :]
        padsz0s = 0
        padsz0e = 0

    if padsz1 < 0:
        Zf = Zf[:, -padsz1s:padsz1e]
        padsz1s = 0
        padsz1e = 0

    Zf = np.pad(
        Zf,
        ((padsz0s, padsz0e), (padsz1s, padsz1e)),
        mode="constant",
        constant_values=(0, 0),
    )

    Zr = sfft.ifft2(sfft.fftshift(Zf))
    return np.abs(Zr)
