import numpy as np
from astropy.io import fits
import pandas as pd
from scipy.optimize import curve_fit

file = fits.open("SPT2147-50-sigmaclipped-g395m-s3d_v2.zip")

header = file[1].header
data = file[1].data
wl_emitted = np.linspace(header["CRVAL3"], header["CRVAL3"]+(header["NAXIS3"]-1)*header["CDELT3"], header["NAXIS3"])
z = 3.7604
wl_emitted = wl_emitted / (1+z)

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return (idx, array[idx])

def avg_3x3(pixelx, pixely):
     return np.nanmean(np.array((data[:, pixelx-1, pixely - 1], data[:, pixelx-1, pixely], data[:, pixelx, pixely], 
           data[:, pixelx, pixely - 1], data[:, pixelx-1, pixely + 1], data[:, pixelx+1, pixely - 1], 
           data[:, pixelx+1, pixely+1], data[:, pixelx+1, pixely], data[:, pixelx, pixely + 1])), axis=0)

def weighted_avg_3x3(pixelx, pixely):
     return np.nanmean(np.array((data[:, pixelx-1, pixely - 1], data[:, pixelx-1, pixely], 3*data[:, pixelx, pixely], 
           data[:, pixelx, pixely - 1], data[:, pixelx-1, pixely + 1], data[:, pixelx+1, pixely - 1], 
           data[:, pixelx+1, pixely+1], data[:, pixelx+1, pixely], data[:, pixelx, pixely + 1])), axis=0)

def avg_5x5(pixelx, pixely):
     return np.nanmean(np.array((data[:, pixelx-1, pixely - 1], data[:, pixelx-1, pixely], data[:, pixelx, pixely], 
           data[:, pixelx, pixely - 1], data[:, pixelx-1, pixely + 1], data[:, pixelx+1, pixely - 1], 
           data[:, pixelx+1, pixely+1], data[:, pixelx+1, pixely], data[:, pixelx, pixely + 1], 
           data[:, pixelx-2, pixely], data[:, pixelx-2, pixely - 1], data[:, pixelx-2, pixely - 2], 
           data[:, pixelx-2, pixely + 1], data[:, pixelx-2, pixely +2], data[:, pixelx-1, pixely +2], 
           data[:, pixelx, pixely +2], data[:, pixelx+1, pixely + 2], data[:, pixelx+2, pixely], 
           data[:, pixelx+2, pixely+2], data[:, pixelx+2, pixely+1], data[:, pixelx+2, pixely-1], 
           data[:, pixelx+2, pixely-2], data[:, pixelx+1, pixely-2], data[:, pixelx, pixely-2], 
           data[:, pixelx-1, pixely-2])), axis=0)

def weighted_avg_5x5(pixelx, pixely):
     return np.nanmean(np.array((data[:, pixelx-1, pixely - 1], data[:, pixelx-1, pixely], data[:, pixelx, pixely], 
           data[:, pixelx, pixely - 1], data[:, pixelx-1, pixely + 1], data[:, pixelx+1, pixely - 1], 
           data[:, pixelx+1, pixely+1], data[:, pixelx+1, pixely], data[:, pixelx, pixely + 1], 
           data[:, pixelx-2, pixely], data[:, pixelx-2, pixely - 1], data[:, pixelx-2, pixely - 2], 
           data[:, pixelx-2, pixely + 1], data[:, pixelx-2, pixely +2], data[:, pixelx-1, pixely +2], 
           data[:, pixelx, pixely +2], data[:, pixelx+1, pixely + 2], data[:, pixelx+2, pixely], 
           data[:, pixelx+2, pixely+2], data[:, pixelx+2, pixely+1], data[:, pixelx+2, pixely-1], 
           data[:, pixelx+2, pixely-2], data[:, pixelx+1, pixely-2], data[:, pixelx, pixely-2], 
           data[:, pixelx-1, pixely-2])), axis=0)

def reduce_cont(pixel):
    rolling_median = ((pd.Series(pixel)).astype('float')).fillna(method='bfill').rolling(100).median()
    return rolling_median

def gaussian3(x, *args):
     a1, b1, c1, a2, b2, c2, a3, b3, c3, C = args
     f1 = a1 * np.exp(-1*((x - b1)**2) / (2*c1**2))
     f2 = a2 * np.exp(-1*((x - b2)**2) / (2*c2**2))
     f3 = a3 * np.exp(-1*((x - b3)**2) / (2*c3**2))
     return f1 + f2 + f3 + C

