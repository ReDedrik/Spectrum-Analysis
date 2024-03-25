import warnings
from data_handling import *
from create_photos import *
warnings.filterwarnings('ignore')

#integrated_spectrum(data)

pixy, pixx = 14, 26 #26, 18 #25, 18 # 25,15 BLOWS
weight = 1
pixel = weighted_avg_5x5(pixx, pixy, weight)
unc = weighted_unc_5x5(pixx, pixy, weight)
idx1, val1 = find_nearest(wl_emitted, 0.66)# 0.65325
idx2, val2 = find_nearest(wl_emitted, 0.69)# 0.66025

max_SII_1 = local_max(pixel, 0.6709, 0.67275)
max_SII_2 = local_max(pixel, 0.67275, 0.675)
C = 0.4

x, y = wl_emitted[idx1:idx2+1], pixel[idx1:idx2+1]
dy = unc[idx1:idx2+1]
#create_spectrum_photos()
#plot_pixel(pixx, pixy)
#plot_avg_3x3(pixx, pixy) 67254

guess_diff = (max_SII_1, 0.00045,
         max_SII_2, 0.00045,
         0, C)

guess_same = (max_SII_1, 0.00045,
         max_SII_2,
         0, C)


bounds_diff = [[max_SII_1-0.1, 0,
           max_SII_2-0.3, 0, -5, 0], 
          [max_SII_1+0.1, 10, 
           max_SII_2 + 0.1, 10, 5, 3]]

bounds_same = [[max_SII_1-0.1, 0,
           max_SII_2-0.3, -5, 0], 
          [max_SII_1+0.1, 10, 
           max_SII_2 + 0.1, 5, 3]]

#bounds_reduced_same = 1

#bounds_reduced_diff = 1


popt_diff, pcov_diff = curve_fit(gaussian2_diff_wid, xdata=x, ydata=y, sigma=dy, p0 = guess_diff, bounds=bounds_diff, maxfev= 1000000)
#step_plot(pixel, idx1, idx2, unc, popt_diff, False)
popt_same, pcov_same = curve_fit(gaussian2_same_wid, xdata=x, ydata=y, sigma=dy, p0 = guess_same, bounds=bounds_same, maxfev= 1000000)


large_step_plot(pixel, idx1, idx2, unc, [popt_diff, popt_same], [pcov_diff, pcov_same], ['Different Widths', 'Same Widths'], [False, True])

cont = reduce_cont(pixel)[idx1:idx2+1]
y = y - cont
guess_reduced = (max_SII_1 - cont[find_nearest(wl_emitted, 0.67166)[0]], 0.00045,
         max_SII_2 - cont[find_nearest(wl_emitted, 0.6731)[0]],
         0, 0)
bounds_reduced = [[max_SII_1-0.1 - cont[find_nearest(wl_emitted, 0.67166)[0]], 0,
           max_SII_2-0.1 - cont[find_nearest(wl_emitted, 0.6731)[0]], -5, 0], 
          [max_SII_1+0.1 - cont[find_nearest(wl_emitted, 0.67166)[0]], 10, 
           max_SII_2 + 0.1 - cont[find_nearest(wl_emitted, 0.6731)[0]], 5, 3]]

guess_reduced_diff = (max_SII_1 - cont[find_nearest(wl_emitted, 0.67166)[0]], 0.00045,
         max_SII_2 - cont[find_nearest(wl_emitted, 0.6731)[0]], 0.00045,
         0, 0)
bounds_reduced_diff = [[max_SII_1-0.1 - cont[find_nearest(wl_emitted, 0.67166)[0]], 0,
           max_SII_2-0.1 - cont[find_nearest(wl_emitted, 0.6731)[0]], 0, -5, 0], 
          [max_SII_1+0.1 - cont[find_nearest(wl_emitted, 0.67166)[0]], 10, 
           max_SII_2 + 0.1 - cont[find_nearest(wl_emitted, 0.6731)[0]], 10, 5, 3]]

popt_reduced_cont, pcov_reduced_cont = curve_fit(gaussian2_same_wid, xdata=x, ydata=y, sigma=dy, p0 = guess_reduced, bounds=bounds_reduced)
popt_reduced_cont_diff, pcov_reduced_cont_diff = curve_fit(gaussian2_diff_wid, xdata=x, ydata=y, sigma=dy, p0 = guess_reduced_diff, bounds=bounds_reduced_diff)

#large_step_plot(pixel - reduce_cont(pixel), idx1, idx2+1, unc, [popt_reduced_cont, popt_reduced_cont_diff], [pcov_reduced_cont, pcov_reduced_cont_diff], ['Same Widths / Reduced Cont.', 'Different Widths / Reduced Cont.'], [True, False])

#integrated_file = fits.open('integrated_SPT2147.fits')
#integrated_data = integrated_file[0].data
#show_img_pixel(integrated_data, pixx, pixy)

# good pixels: (11, 26-28), (25, 18), (20, 14-17), (28, 15-19)
# try to fit every pixel, and if it is bad then cut it out