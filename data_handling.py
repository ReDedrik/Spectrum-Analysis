import numpy as np
from astropy.io import fits
import pandas as pd
from scipy.optimize import curve_fit
import scienceplots
import matplotlib.pyplot as plt
plt.style.use('science')
plt.rc('lines', linewidth=2)
file = fits.open("SPT2147-50-sigmaclipped-g395m-s3d_v2.zip")

header = file[1].header
data = file[1].data
uncertainties = file[2].data
wl_emitted = np.linspace(header["CRVAL3"], header["CRVAL3"]+(header["NAXIS3"]-1)*header["CDELT3"], header["NAXIS3"])
z = 3.7604
wl_emitted = wl_emitted / (1+z)

def quickselect(L, k):
     x = L[0]
     L1, L2, L3 = [], [], []
     for i in L:
          if i < x:
               L1.append(i)
          elif i == x:
               L2.append(i)
          else:
               L3.append(i)
     if k <= len(L1):
          return quickselect(L1, k)
     elif k > len(L1) + len(L2):
          return quickselect(L3, k - len(L1) - len(L2))
     return x

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
     if weight > 25:
          weight = 25
     main_weight, sub_weight = weight / 25, (1 - (weight / 25)) / 24
     return np.nansum(sub_weight * np.array((data[:, pixelx-1, pixely - 1], data[:, pixelx-1, pixely], (24 * main_weight / (1 - main_weight)) * data[:, pixelx, pixely], 
           data[:, pixelx, pixely - 1], data[:, pixelx-1, pixely + 1], data[:, pixelx+1, pixely - 1], 
           data[:, pixelx+1, pixely+1], data[:, pixelx+1, pixely], data[:, pixelx, pixely + 1], 
           data[:, pixelx-2, pixely], data[:, pixelx-2, pixely - 1], data[:, pixelx-2, pixely - 2], 
           data[:, pixelx-2, pixely + 1], data[:, pixelx-2, pixely +2], data[:, pixelx-1, pixely +2], 
           data[:, pixelx, pixely +2], data[:, pixelx+1, pixely + 2], data[:, pixelx+2, pixely], 
           data[:, pixelx+2, pixely+2], data[:, pixelx+2, pixely+1], data[:, pixelx+2, pixely-1], 
           data[:, pixelx+2, pixely-2], data[:, pixelx+1, pixely-2], data[:, pixelx, pixely-2], 
           data[:, pixelx-1, pixely-2])), axis=0)

def weighted_unc_5x5(pixelx, pixely, weight):
     if weight > 25:
          weight = 25
     main_weight, sub_weight = weight / 25, (1 - (weight / 25)) / 24
     return np.nansum(sub_weight * np.array((uncertainties[:, pixelx-1, pixely - 1], uncertainties[:, pixelx-1, pixely], (24 * main_weight / (1 - main_weight)) * uncertainties[:, pixelx, pixely], 
           uncertainties[:, pixelx, pixely - 1], uncertainties[:, pixelx-1, pixely + 1], uncertainties[:, pixelx+1, pixely - 1], 
           uncertainties[:, pixelx+1, pixely+1], uncertainties[:, pixelx+1, pixely], uncertainties[:, pixelx, pixely + 1], 
           uncertainties[:, pixelx-2, pixely], uncertainties[:, pixelx-2, pixely - 1], uncertainties[:, pixelx-2, pixely - 2], 
           uncertainties[:, pixelx-2, pixely + 1], uncertainties[:, pixelx-2, pixely +2], uncertainties[:, pixelx-1, pixely +2], 
           uncertainties[:, pixelx, pixely +2], uncertainties[:, pixelx+1, pixely + 2], uncertainties[:, pixelx+2, pixely], 
           uncertainties[:, pixelx+2, pixely+2], uncertainties[:, pixelx+2, pixely+1], uncertainties[:, pixelx+2, pixely-1], 
           uncertainties[:, pixelx+2, pixely-2], uncertainties[:, pixelx+1, pixely-2], uncertainties[:, pixelx, pixely-2], 
           uncertainties[:, pixelx-1, pixely-2])), axis=0)

def local_max(pixel, x1, x2):
     return np.nanmax(pixel[find_nearest(wl_emitted, x1)[0]:find_nearest(wl_emitted, x2)[0]])

def reduce_cont(pixel):
    rolling_median = ((pd.Series(pixel)).astype('float')).fillna(method='bfill').rolling(100).median()
    return rolling_median

def gaussian3(x, *args):
     amp1, width1, amp2, width2, amp3, pos3, width3, m, C = args
     amp1 = amp1 - m*x - C
     amp2 = amp2 - m*x - C
     f1 = amp1 * np.exp(-1*((x - 0.67166)**2) / (2*width1**2))# n2-1 is 0.654985
     f2 = amp2 * np.exp(-1*((x - 0.6731)**2) / (2*width1**2))# halpha is 0.65641
     f3 = amp3 * np.exp(-1*((x - pos3)**2) / (2*width3**2)) # n2-2 is 0.658528
     broad = 0 # a4 * np.exp(-1*((x - b4)**2) / (2*c4**2))
     return f1 + f2 + 0 + broad + C + m*x

def gaussian2_diff_wid(x, *args):
     amp1, width1, amp2, width2, m, C = args
     amp1 = amp1 - m*x - C
     amp2 = amp2 - m*x - C
     f1 = amp1 * np.exp(-1*((x - 0.67166)**2) / (2*width1**2))
     f2 = amp2 * np.exp(-1*((x - 0.6731)**2) / (2*width2**2))
     return f1 + f2 + m*x + C

def gaussian2_same_wid(x, *args):
     amp1, width1, amp2, m, C = args
     amp1 = amp1 - m*x - C
     amp2 = amp2 - m*x - C
     f1 = amp1 * np.exp(-1*((x - 0.67166)**2) / (2*width1**2))
     f2 = amp2 * np.exp(-1*((x - 0.6731)**2) / (2*width1**2))
     return f1 + f2 + m*x + C



# make function to subtract larger continuum instead of narrow