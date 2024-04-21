import warnings
from data_handling import *
from create_photos import *
warnings.filterwarnings('ignore')

#integrated_spectrum(data)

pixx, pixy = 14, 26 #26, 18 #25, 18 # 25,15 BLOWS
weight = 1
pixel = weighted_avg_5x5(pixy, pixx, weight)
unc = weighted_unc_5x5(pixy, pixx, weight)
idx1, val1 = find_nearest(wl_emitted, 0.661)# 0.65325
idx2, val2 = find_nearest(wl_emitted, 0.69)# 0.66025

max_SII_1 = local_max(pixel, 0.6709, 0.67275)
max_SII_2 = local_max(pixel, 0.67275, 0.675)
C = 0.4

x, y = wl_emitted[idx1:idx2+1], pixel[idx1:idx2+1]
dy = unc[idx1:idx2+1]

m_guess = 0.01
guess_same = (0.4, 0.0003,
         0.7,
         m_guess, C, z)


bounds_same = [[0, 0,
           0, -5, 0, 3.7601], 
          [10, 0.0005, 
           10, 10, 5, 3.7607]]


#popt_same, pcov_same = curve_fit(gaussian2_same_wid, xdata=x, ydata=y, sigma=dy, p0 = guess_same, bounds=bounds_same, maxfev= 1000000)

#large_step_plot(pixel, idx1, idx2, unc, [popt_diff, popt_same], [pcov_diff, pcov_same], ['Different Widths', 'Same Widths'], [False, True])


indices = index_finder()
pix1 = Pixel(14, 26)
pix1.average_values()
#pix1.plot_pixel()

pix2 = Pixel(14, 26)
#pix2.average_values()
#pix2.plot_pixel()

pix3 = Pixel(34, 30)

indxs=[indices[0] - 50, indices[1] + 40]
popt1, pcov1 = pix1.fit_pixel(guess_same, bounds_same, [idx1, idx2])
popt2, pcov2 = pix2.fit_pixel(guess_same, bounds_same, [idx1, idx2])

#pix1.plot_spectrum(indxs = [idx1, idx2], fit_params=[popt1, pcov1])
pix2.plot_spectrum(indxs = [idx1, idx2])
pix3.average_values()
pix3.fit_pixel(guess_same, bounds_same, [idx1, idx2])
pix3.plot_spectrum(indxs = [idx1, idx2])

# good pixels: (11, 26-28), (25, 18), (20, 14-17), (28, 15-19)
# try to fit every pixel, and if it is bad then cut it out
# fit redshift to correct SII locations

# fit halpha and n2