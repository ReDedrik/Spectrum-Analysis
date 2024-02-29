import warnings
from not_main import *
from create_photos import *
warnings.filterwarnings('ignore')



pixx, pixy = 25, 18
pixel = avg_5x5(pixx, pixy)
idx1, val1 = find_nearest(wl_emitted, 0.65325)
idx2, val2 = find_nearest(wl_emitted, 0.66025)

x, y = wl_emitted[idx1:idx2+1], pixel[idx1:idx2+1]
#create_spectrum_photos()
#plot_pixel(pixx, pixy)
#plot_avg_3x3(pixx, pixy)
guess = (0.77, 0.6549, 0.0004, 2, 0.65635, 0.00120, 1.45, 0.65845, 0.00070, 0.2) 
popt, pcov = curve_fit(gaussian3, xdata=x, ydata=y, p0 = guess)
print(popt)
plt.step(wl_emitted[idx1:idx2], avg_5x5(pixx, pixy)[idx1:idx2], where='mid')
plt.plot(np.linspace(wl_emitted[idx1], wl_emitted[idx2], 100000), gaussian3(np.linspace(wl_emitted[idx1], wl_emitted[idx2], 100000), *popt), ls='--')

#pixel_comparison(data[:, pixx, pixy], avg_3x3(pixx, pixy), weighted_avg_3x3(pixx, pixy), loc = (pixx, pixy), xlims=(0.64, 0.68), ylims=(-0.1, 2))


plt.show()
# try 5x5 avg
# nanmedian
# should give higher weights to center pixel when taking surrounding average?
# subtract continuum for individual pixels, or integrated photo continuum? 
# ISSUE: when taking average of surrounding pixels on pixels that have lots of NaN's, screws up data
# subtract continuum from each individual 
# fitting gaussian to lines
# good pixels: (11, 26-28), (25, 18), (20, 14-17), (28, 15-19)

'''
need to fit curves and then can integrate those fitted curves from specified ranges
'''