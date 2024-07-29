import warnings
from data_handling import *
from create_photos import *
warnings.filterwarnings('ignore')

#integrated_spectrum(data)

pixx, pixy = 14, 26
weight = 1
pixel = weighted_avg_5x5(pixy, pixx, weight)
unc = weighted_unc_5x5(pixy, pixx, weight)
idx1, val1 = find_nearest(wl_emitted, 0.667)# 0.65325
idx2, val2 = find_nearest(wl_emitted, 0.678)# 0.66025   


#idx1, val1 = find_nearest(wl_emitted, 0.653)# 0.65325
#idx2, val2 = find_nearest(wl_emitted, 0.665)# 0.66025

C = 2

x, y = wl_emitted[idx1:idx2+1], pixel[idx1:idx2+1]
dy = unc[idx1:idx2+1]

m_guess = 0.01
guess_same = (0.5, 0.0003,
               0.6, C, z)


bounds_same = [[0, 0, 0, 0, 1], 
          [10, 0.1, 10, 10, 5.7618]]


#popt_same, pcov_same = curve_fit(gaussian2_same_wid, xdata=x, ydata=y, sigma=dy, p0 = guess_same, bounds=bounds_same, maxfev= 1000000)

#large_step_plot(pixel, idx1, idx2, unc, [popt_diff, popt_same], [pcov_diff, pcov_same], ['Different Widths', 'Same Widths'], [False, True])


pix3 = Pixel(28, 12)


pix3.fit_pixel(guess_same, bounds_same, [idx1, idx2])
pix3.plot_spectrum(indxs = [idx1, idx2], show=True)

create_spectrum_photos()

# good pixels: (11, 26-28), (25, 18), (20, 14-17), (28, 15-19)
# try to fit every pixel, and if it is bad then cut it out
# fit redshift to correct SII locations

# fit halpha and n2