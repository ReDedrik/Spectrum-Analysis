import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.image as mpimg
from matplotlib import animation
#import astropy.units as units
#from astropy.stats import SigmaClip
#from astropy.visualization import simple_norm
#from astropy.convolution import convolve
from astropy.io import fits
#from astropy.stats import sigma_clipped_stats


file = fits.open("SPT2147-50-sigmaclipped-g395m-s3d_v2.zip")

header = file[1].header
data = file[1].data
wl = np.linspace(header["CRVAL3"], header["CRVAL3"]+(header["NAXIS3"]-1)*header["CDELT3"], header["NAXIS3"])
redshift = 3.7604

pixx = 22


def plot_pixel(pixelx, pixely, xlims=(min(wl), max(wl))):
    plt.figure(figsize=(7,4))
    pixel = data[:, pixelx, pixely]
    plt.plot(wl, pixel)
    plt.plot(wl, reduce_cont(pixel), color='orange')
    plt.xticks(np.arange(min(wl), max(wl), 0.2))
    #plt.xlim(min(wl) + 14*header["CDELT3"], max(wl) - 12*header["CDELT3"])
    plt.title(f"Spectrum of Pixel ({pixelx}, {pixely})")
    plt.xlim(xlims)
    plt.ylim(-5, 5)
    plt.minorticks_on()
    plt.show()

def plot_avg_3x3(pixelx, pixely, xlims=(min(wl), max(wl))):
    plt.figure(figsize=(7,4))
    pixel = data[:, pixelx, pixely]
    plt.plot(wl, pixel)
    plt.plot(wl, reduce_cont(pixel), color='orange')
    plt.xticks(np.arange(min(wl), max(wl), 0.2))
    #plt.xlim(min(wl) + 14*header["CDELT3"], max(wl) - 12*header["CDELT3"])
    plt.title(f"Spectrum of Pixel ({pixelx}, {pixely})")
    plt.xlim(xlims)
    plt.ylim(-5, 5)
    plt.minorticks_on()

def avg_3x3(pixelx, pixely):
     return np.nanmean(np.array((data[:, pixelx-1, pixely - 1], data[:, pixelx-1, pixely], data[:, pixelx, pixely], 
           data[:, pixelx, pixely - 1], data[:, pixelx-1, pixely + 1], data[:, pixelx+1, pixely - 1], 
           data[:, pixelx+1, pixely+1], data[:, pixelx+1, pixely], data[:, pixelx, pixely + 1])), axis=0)

def reduce_cont(pixel):
    rolling_median = ((pd.Series(pixel)).astype('float')).fillna(method='bfill').rolling(100).median()
    return rolling_median

def reduce_cont_integrated(pixel):
    pass

def create_spectrum_photos():
    for i in range(np.shape(data)[1]):
        for j in range(np.shape(data)[2]):
            if not np.isnan(data[:, i, j]).all():
                plot_avg_3x3(i, j)
                if not os.path.exists(f"pixels/{i}_pixels"):
                    os.mkdir(f"pixels/{i}_pixels")
                plt.savefig(f'pixels/{i}_pixels/pixel_({i}, {j}).png')
                plt.close();


#create_spectrum_photos()

def init():
    imobj.set_data(np.zeros(np.shape(data[:, pixx, :])))
    return imobj



plot_pixel(25, 25)

fig = plt.figure()
ax = plt.gca()
imobj = ax.imshow(np.zeros(np.shape(data[:, pixx, :])), origin='lower', alpha=1.0, zorder=1, aspect=1)

#anim = animation.FuncAnimation(fig, update, int_func=init, repeat=True, frames=range()))
'''

reduce_cont(data[:, pixx, pixy])
plot_pixel(data[:, pixx, pixy])
plot_pixel(avg_3x3(pixx, pixy))


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
plt.plot(wl, reduce_cont(data[:, pixx, pixy]), color='orange')

plt.figure(figsize=(5, 5))
plt.plot()
'''