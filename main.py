import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
#from matplotlib import animation
#import astropy.units as units
#from astropy.stats import SigmaClip
#from astropy.visualization import simple_norm
#from astropy.convolution import convolve
from astropy.io import fits
import os
import shutil

#from astropy.stats import sigma_clipped_stats

file = fits.open("SPT2147-50-sigmaclipped-g395m-s3d_v2.zip")

header = file[1].header
data = file[1].data
wl_emitted = np.linspace(header["CRVAL3"], header["CRVAL3"]+(header["NAXIS3"]-1)*header["CDELT3"], header["NAXIS3"])
z = 3.7604
wl_emitted = wl_emitted / (1+z)

# working
def pixel_comparison(*args, loc=('-', '-'), xlims=(min(wl_emitted), max(wl_emitted)), ylims=(-1, 3)):
    fig, ax = plt.subplots(len(args), sharey = True, figsize=(7, len(args) * 4))
    for i in range(len(args)):
        ax[i].plot(wl_emitted, args[i])
        ax[i].plot(wl_emitted, reduce_cont(args[i]), color='orange')
        ax[i].set_xticks(np.linspace(xlims[0], xlims[1], 10))
        ax[i].set_title(f'Pixel Iter. {i}')
        ax[i].set_xlim(xlims)
        ax[i].set_ylim(ylims)
        ax[i].minorticks_on()
        ax[i].axvline(0.6562, linestyle='dotted', color='red')
        ax[i].axvline(0.6583, linestyle='dotted', color='red')
        ax[i].axvline(0.9531, linestyle='dotted', color='red')
        ax[i].axvline(1.083, linestyle='dotted', color='red')   
        ax[i].axvline(0.9068, linestyle='dotted', color='red')
        ax[i].axvline(0.6730, linestyle='dotted', color='red')
    #ax.vlines(0.6562, 0.6583, 0.9531, 1.083, 0.9068, 0.6730)
    fig.suptitle(f'Pixel at ({loc[0]}, {loc[1]})')
    plt.show()
        
# ALSO NOT WORKING RN
def plot_pixel(pixelx, pixely, xlims=(min(wl_emitted), max(wl_emitted))):
    plt.figure(figsize=(7,4))
    pixel = data[:, pixelx, pixely]
    plt.plot(wl_emitted, pixel)
    plt.plot(wl_emitted, reduce_cont(pixel), color='orange')
    plt.xticks(np.arange(min(wl_emitted), max(wl_emitted), 0.02))
    plt.title(f"Spectrum of Pixel ({pixelx}, {pixely})")
    plt.xlim(xlims)
    plt.ylim(-5, 5)
    plt.minorticks_on()
    #plt.show()

# NOT WORKING RN
def plot_avg_3x3(pixelx, pixely, xlims=(min(wl_emitted), max(wl_emitted))):
    plt.figure(figsize=(7,4))
    pixel = data[:, pixelx, pixely]
    plt.plot(wl_emitted, pixel)
    plt.plot(wl_emitted, reduce_cont(pixel), color='orange')
    plt.xticks(np.arange(min(wl_emitted), max(wl_emitted), 0.02))
    plt.title(f"Spectrum of Pixel ({pixelx}, {pixely})")
    plt.xlim(xlims)
    plt.ylim(-5, 5)
    plt.minorticks_on()
    #plt.show()

def clear_photos():
    sub_dir = [i[0] for i in os.walk("pixels")]
    print(sub_dir)
    for i in sub_dir[1:]:
        shutil.rmtree(i, ignore_errors=True)


def create_spectrum_photos():
    clear_photos()
    for i in range(np.shape(data)[1]):
        for j in range(np.shape(data)[2]):
            if not np.isnan(data[:, i, j]).all():
                #plot_avg_3x3(i, j)
                pixel_comparison(data[:, i, j], avg_5x5(i, j), loc=(i, j))
                if not os.path.exists(f"pixels/{i}_pixels"):
                    os.mkdir(f"pixels/{i}_pixels")
                plt.savefig(f'pixels/{i}_pixels/pixel_({i}, {j}).png')
                plt.close();


def avg_3x3(pixelx, pixely):
     return np.nanmean(np.array((data[:, pixelx-1, pixely - 1], data[:, pixelx-1, pixely], data[:, pixelx, pixely], 
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


def reduce_cont(pixel):
    rolling_median = ((pd.Series(pixel)).astype('float')).fillna(method='bfill').rolling(100).median()
    return rolling_median

def reduce_cont_integrated(pixel):
    pass

pixx, pixy = 25, 18

#create_spectrum_photos()
#plot_pixel(pixx, pixy)
#plot_avg_3x3(pixx, pixy)

pixel_comparison(data[:, pixx, pixy],avg_3x3(pixx, pixy), avg_5x5(pixx, pixy), loc = (pixx, pixy), xlims=(0.64, 0.68), ylims=(-0.1, 2))


# This code subtracted the continuum but kinda sucks and i dont like it
# but best i currently have
'''
arr1 = avg_3x3(pixx, pixy)

producand1 = np.full_like(data[:, pixx, pixy], -1)

reduced_cont = reduce_cont(data[:, pixx, pixy])
producand2 = np.array(reduced_cont)

arr2 = np.nanprod(np.dstack((producand1, producand2)), axis=2)
stacked_array = np.dstack((arr1, arr2))
nansumed = np.nansum(stacked_array, 2)
plot_pixel(nansumed[0])

plot_pixel(data[:, pixx, pixy], xlims=(3.06, 3.2))

plt.figure(figsize=(7,4))
plt.plot(wl_emitted, reduce_cont(data[:, pixx, pixy]), color='orange')

plt.figure(figsize=(5, 5))
plt.plot()
'''

# try 5x5 avg
# nanmedian
# should give higher weights to center pixel when taking surrounding average?
# subtract continuum for individual pixels, or integrated photo continuum? 
# ISSUE: when taking average of surrounding pixels on pixels that have lots of NaN's, screws up data
# subtract continuum from each individual 
# fitting gaussian to lines
# good pixels: (11, 26-28), (25, 18), (20, 14-17), (28, 15-19)