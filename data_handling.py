import numpy as np
from astropy.io import fits
import pandas as pd
from scipy.optimize import curve_fit
import scienceplots
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.ticker as ticker
from scipy.stats import chisquare
from math import factorial
import vorbin


plt.style.use('science')
plt.rc('lines', linewidth=2)
plt.rc('text', usetex=True)

file = fits.open("SPT2147-50-sigmaclipped-g395m-s3d_v2.zip")

header = file[1].header
data = file[1].data
uncertainties = file[2].data
wl_obs = np.linspace(header["CRVAL3"], header["CRVAL3"]+(header["NAXIS3"]-1)*header["CDELT3"], header["NAXIS3"])
z = 3.7604
wl_emitted = wl_obs / (1+z)
print(file.info())

class Pixel:
     def __init__(self, pixx, pixy):
          self.x = pixx
          self.y = pixy
          self.z = 3.7604
          self.pixel = data[:, self.y, self.x]
          self.wl_emitted = wl_emitted
          self.unc = uncertainties[:, self.y, self.x]
          self.fontsize = 22
          self.popt = None
          self.pcov = None

     def __str__(self):
          return f'({self.x}, {self.y})'

     def return_spectrum(self):
          pass
     
     def average_values(self, undo = False):
          if undo:
               self.pixel = self.pixel = data[:, self.y, self.x]
               return
          padded_data = np.pad(data, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=np.nan)
          self.pixel = np.nanmean(padded_data[:, self.x - 1 : self.x + 1, self.y - 1: self.y + 1], axis=(1, 2))
          

     def fit_pixel(self, guess, bounds, indxs = []):
          idx1, idx2 = indxs
          self.popt, self.pcov = curve_fit(gaussian2, xdata=wl_obs[idx1:idx2], ydata=self.pixel[idx1:idx2], sigma=self.unc[idx1:idx2], p0 = guess, bounds=bounds, maxfev= 10000)
          print(self.popt)
          return self.popt, self.pcov

     def plot_spectrum(self, indxs = [], show = True):
          idx1, idx2 = indxs
          #smoothed_curve = savitzky_golay(self.pixel[idx1:idx2], 29, 3)
          if len(self.popt) != 0:
               print('poop')
               self.z = self.popt[-1]
               #self.z = 3.7604
               self.wl_emitted = wl_obs / (1+self.z)
          fig, axes = plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={'height_ratios' : [2, 1], 'hspace' : 0.05}, sharex=True)
          axes[0].step(self.wl_emitted[idx1:idx2], self.pixel[idx1:idx2], where='mid')
          axes[0].fill_between(self.wl_emitted[idx1:idx2], self.pixel[idx1:idx2] - self.unc[idx1:idx2], self.pixel[idx1:idx2] + self.unc[idx1:idx2], alpha=0.2)
          axes[0].plot(self.wl_emitted, gaussian2(wl_obs, *self.popt), ls='--', label='Fitted Curve', color='mediumseagreen', zorder=6)
          #axes[0].plot(self.wl_emitted[idx1:idx2], smoothed_curve, label = 'SG-Curve')
          
          axes[0].set_title(f'({self.x}, {self.y})', fontsize = self.fontsize)
          axes[0].minorticks_on()
          axes[0].axvline(0.671829, linestyle='--', color='gray', alpha=0.6, linewidth=1)
          axes[0].axvline(0.673267, linestyle='--', color='gray', alpha = 0.6, linewidth=1)
          axes[0].set_xlim(self.wl_emitted[idx1], self.wl_emitted[idx2])
          axes[0].tick_params(axis='both', labelsize= 16)
          #axes[0].set_ylim(0.8, 3)
          residuals = self.pixel[idx1:idx2] - gaussian2(wl_obs[idx1:idx2], *self.popt)
          axes[1].scatter(self.wl_emitted[idx1:idx2], residuals, color='black', zorder=5)
          axes[1].axhline(0, alpha=0.4, color='gray')
          axes[1].fill_between(self.wl_emitted[idx1:idx2], residuals - self.unc[idx1:idx2], residuals + self.unc[idx1:idx2], alpha=0.2)
          axes[1].set_xlim(self.wl_emitted[idx1], self.wl_emitted[idx2-1])
          axes[1].tick_params(axis='both', labelsize= 16)
          axes[1].axvline(0.671829, linestyle='--', color='gray', alpha=0.6, linewidth=1)
          axes[1].axvline(0.673267, linestyle='--', color='gray', alpha = 0.6, linewidth=1)
          print(self.popt[1])
          fig.legend()
          if show:
               plt.show()



     def plot_pixel(self, save = False):
          int_data = integrated_data()
          plt.figure(figsize=(7, 7))
          plt.imshow(int_data, origin='lower')
          plt.scatter([self.x], [self.y], color='red', s=10)
          plt.colorbar()
          if not save:
               plt.show()
               return
          plt.savefig('C:/Users/redma/Downloads/SII_pixel')


def integrated_data():
     integrated_file = fits.open('integrated_SPT2147.fits')
     integrated_data = integrated_file[0].data
     for i in range(len(integrated_data)):
          for j in range(len(integrated_data[i])):
               if integrated_data[i][j] == 0:
                    integrated_data[i][j] = np.nan
     return integrated_data

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

def gaussian2(x, *args):
     amp1, width1, amp2, C, z = args
     f1 = amp1 * np.exp(-1*((x - 0.671829 * (1+z))**2) / (2*(width1)**2))
     f2 = amp2 * np.exp(-1*((x - 0.673267 * (1+z))**2) / (2*(width1)**2))
     return f1 + f2 + C

def gaussian3(x, *args):
     amp1, width1, amp2, C, z = args
     #f1 = amp1 * np.exp(-1*(((x - 0.6548050) * (1+z))**2) / (2*(width1 * (1+z))**2))
     #f2 = amp2 * np.exp(-1*(((x - 0.6562819) * (1+z))**2) / (2*(width2 * (1+z))**2))
     #f3 = amp3 * np.exp(-1*(((x - 0.6583460) * (1+z))**2) / (2*(width1 * (1+z))**2))
     f1 = amp1 / 2.8 * np.exp(-1*((x - 0.654985*(1+z))**2) / (2*(width1)**2))
     f2 = amp2 * np.exp(-1*((x - 0.656461*(1+z))**2) / (2*(width1)**2))
     f3 = amp1 * np.exp(-1*((x - 0.658528*(1+z))**2) / (2*(width1)**2))
     return f1 + f2 + f3 + C

def integrated_spectrum(data):
     wl_file = np.empty((np.shape(data)[1], np.shape(data)[2]))
     for i in range(np.shape(data)[1]):
          for j in range(np.shape(data)[2]):
               wl_file[i, j] = np.nansum(data[:, i , j], axis=0)
     hdu = fits.PrimaryHDU(wl_file)
     hdu.writeto('integrated_SPT2147.fits')
     
def index_finder():
     sII_1 = find_nearest(wl_emitted, 0.671644)[0]
     sII_2 = find_nearest(wl_emitted, 0.673081)[0]
     return sII_1, sII_2

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int64(window_size))
        order = np.abs(np.int64(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

# make function to subtract larger continuum instead of narrow
     



'''

pick a pixel
if S/N (integrated gaussians / uncertainties) below 3, reject pixel
fit that pixel in observed space (fit for z)
using z, convert to emitted space
if fit good, keep it

to generalize to any photo:
1. aperture that ho
2. convert to wl_emitted
3. find high(ish) detection of line
4. leave wl_obs and fit every pixels redshift based on known indexes of lines

'''