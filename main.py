import warnings
from data_handling import *
from create_photos import *
warnings.filterwarnings('ignore')



pixx, pixy = 28, 19 #26, 18 #25, 18 # 25,15 BLOWS
weight = 1
pixel = weighted_avg_5x5(pixx, pixy, weight)
unc = weighted_unc_5x5(pixx, pixy, weight)
idx1, val1 = find_nearest(wl_emitted, 0.66)# 0.65325
idx2, val2 = find_nearest(wl_emitted, 0.685)# 0.66025

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

#guess_reduced_same = (max)

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

#popt_reduced_same, pcov_reduced_same = curve_fit(gaussian2_same_wid, xdata=x, ydata=y, sigma=dy, p0 = guess_reduced_same, bounds=bounds_reduced_same, maxfev= 1000000)

#popt_reduced_diff, pcov_reduced_diff = curve_fit(gaussian2_diff_wid, xdata=x, ydata=y, sigma=dy, p0 = guess_reduced_diff, bounds=bounds_reduced_diff, maxfev= 1000000)
print(popt_same)
#step_plot(pixel, idx1, idx2, unc, popt_same, True)

large_step_plot(pixel, idx1, idx2, unc, [popt_diff, popt_same], ['Different Widths', 'Same Widths'], [False, True])

#pixel_comparison(pixel, pixel)
#lambda6716, lambda6731 = popt[0], popt[3]
#m, C = popt[-2], popt[-1]
# should i be restricting the amplitude so much
# this value?
#print(lambda6716 / lambda6731)
# this value?
#print((lambda6716 - pixel[idx1] * m - C) / (lambda6731 - pixel[idx1] * m - C))
# or a different value without taking into account m and C
plt.show()

#show_img_pixel(data[50], pixx, pixy)

#plot residuals, plot different parameters (change widths, fit more continuum, wider range)
# good pixels: (11, 26-28), (25, 18), (20, 14-17), (28, 15-19)