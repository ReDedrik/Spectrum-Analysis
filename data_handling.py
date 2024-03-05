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

def weighted_avg_5x5(pixelx, pixely, weight):
     return np.nanmean(np.array((data[:, pixelx-1, pixely - 1], data[:, pixelx-1, pixely], weight * data[:, pixelx, pixely], 
           data[:, pixelx, pixely - 1], data[:, pixelx-1, pixely + 1], data[:, pixelx+1, pixely - 1], 
           data[:, pixelx+1, pixely+1], data[:, pixelx+1, pixely], data[:, pixelx, pixely + 1], 
           data[:, pixelx-2, pixely], data[:, pixelx-2, pixely - 1], data[:, pixelx-2, pixely - 2], 
           data[:, pixelx-2, pixely + 1], data[:, pixelx-2, pixely +2], data[:, pixelx-1, pixely +2], 
           data[:, pixelx, pixely +2], data[:, pixelx+1, pixely + 2], data[:, pixelx+2, pixely], 
           data[:, pixelx+2, pixely+2], data[:, pixelx+2, pixely+1], data[:, pixelx+2, pixely-1], 
           data[:, pixelx+2, pixely-2], data[:, pixelx+1, pixely-2], data[:, pixelx, pixely-2], 
           data[:, pixelx-1, pixely-2])), axis=0)

def local_max(pixel, x1, x2):
     return np.nanmax(pixel[find_nearest(wl_emitted, x1)[0]:find_nearest(wl_emitted, x2)[0]])

def reduce_cont(pixel):
    rolling_median = ((pd.Series(pixel)).astype('float')).fillna(method='bfill').rolling(100).median()
    return rolling_median

def gaussian3(x, *args):
     a1, b1, c1, a2, b2, c2, a3, b3, c3, m, C = args
     a1 = a1 - m*x + C
     a2 = a2 - m*x + C
     f1 = a1 * np.exp(-1*((x - 0.67166)**2) / (2*c1**2))# n2-1 is 0.654985
     f2 = a2 * np.exp(-1*((x - 0.6731)**2) / (2*c1**2))# halpha is 0.65641
     f3 = a3 * np.exp(-1*((x - b3)**2) / (2*c3**2)) # n2-2 is 0.658528
     broad = 0 # a4 * np.exp(-1*((x - b4)**2) / (2*c4**2))
     return f1 + f2 + f3 + broad + C + m*x

def gaussian2(x, *args):
     a1, b1, c1, a2, b2, c2, m, C = args
     f1 = a1 * np.exp(-1*((x - 0.67166)**2) / (2*c1**2))
     f2 = a2 * np.exp(-1*((x - 0.6731)**2) / (2*c1**2))
     return f1 + f2 + m*x + C