import warnings
from data_handling import *
from create_photos import *
warnings.filterwarnings('ignore')



pixx, pixy = 25, 18 #26, 18 #25, 18 # 25,15 BLOWS
pixel = weighted_avg_5x5(pixx, pixy, 2)
unc = weighted_unc_5x5(pixx, pixy, 2)
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
guess = (np.nanmax(pixel[find_nearest(wl_emitted, 0.65433)[0]:find_nearest(wl_emitted, 0.65531)[0]]), 1, 0.0004, 
         np.nanmax(pixel[find_nearest(wl_emitted, 0.65550)[0]:find_nearest(wl_emitted, 0.65740)[0]]), 1, 0.00120, 
         np.nanmax(pixel[find_nearest(wl_emitted, 0.65770)[0]:find_nearest(wl_emitted, 0.65975)[0]]), 1, 0.00070, 
         1, 1, 1, 0, 0.3) 

guess = (np.nanmax(pixel[find_nearest(wl_emitted, 0.6709)[0]:find_nearest(wl_emitted, 0.6726)[0]]), 1, 0.0002,
         np.nanmax(pixel[find_nearest(wl_emitted, 0.6726)[0]:find_nearest(wl_emitted, 0.675)[0]]), 1, 0.0002, 
         0, 0.2)

guess = (max_SII_1, 1, 0.0002,
         max_SII_2, 1, 0.0002, 
         -0.2, 0.67254, 0.0002,
         0, C)

bounds = [[max_SII_1 - guess[-1] - 0.003,  0,  0,
           max_SII_2 - guess[-1] - 0.003,  0,  0,
           -10, 0.67253, 0.00001, -5, 0], 
          [max_SII_1 - guess[-1] + 0.1, 10, 10, 
           max_SII_2 - guess[-1] + 0.1, 10, 10, 
           0,  0.67255,   0.001,  5, 1]]

bounds = [[max_SII_1-0.001,  0,  0,
           max_SII_2-0.001,  0,  0,
           -10, 0.67253, 0.00006, -5, 0], 
          [max_SII_1+0.001, 10, 10, 
           max_SII_2 + 0.001, 10, 10, 
           0,  0.67255,   0.001,  5, 1]]

popt, pcov = curve_fit(gaussian3, xdata=x, ydata=y, sigma=dy, p0 = guess, bounds=bounds, maxfev= 1000000)
print(popt)
plt.step(wl_emitted[idx1:idx2], pixel[idx1:idx2], where='pre')
plt.fill_between(wl_emitted[idx1:idx2], pixel[idx1:idx2] - unc[idx1:idx2], pixel[idx1:idx2] + unc[idx1:idx2], alpha=0.2)
plt.plot(np.linspace(wl_emitted[idx1], wl_emitted[idx2], 10000), gaussian3(np.linspace(wl_emitted[idx1], wl_emitted[idx2], 10000), *popt), ls='--')

lambda6716, lambda6731 = popt[0], popt[3]
m, C = popt[-2], popt[-1]
# should i be restricting the amplitude so much
# this value?
print(lambda6716 / lambda6731)
# this value?
print((lambda6716 - pixel[idx1] * m - C) / (lambda6731 - pixel[idx1] * m - C))
# or a different value without taking into account m and C
plt.show()

#show_img_pixel(data[50], pixx, pixy)


# good pixels: (11, 26-28), (25, 18), (20, 14-17), (28, 15-19)