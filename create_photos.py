import shutil
import numpy as np
import os
from data_handling import *

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
        ax[i].plot(wl_emitted, reduce_cont(args[i]))
        ax[i].axvline(0.6562, linestyle='dotted', color='red')
        ax[i].axvline(0.6583, linestyle='dotted', color='red')
        ax[i].axvline(0.9531, linestyle='dotted', color='red')
        ax[i].axvline(1.083, linestyle='dotted', color='red')   
        ax[i].axvline(0.9068, linestyle='dotted', color='red')
        ax[i].axvline(0.6730, linestyle='dotted', color='red')
    #ax.vlines(0.6562, 0.6583, 0.9531, 1.083, 0.9068, 0.6730)
    fig.suptitle(f'Pixel at ({loc[0]}, {loc[1]})')
    plt.show()

def clear_photos():
    sub_dir = [i[0] for i in os.walk("pixels")]
    print(sub_dir)
    for i in sub_dir[1:]:
        shutil.rmtree(i, ignore_errors=True)

def create_spectrum_photos():
    clear_photos()
    idx1, val1 = find_nearest(wl_emitted, 0.664)# 0.65325
    idx2, val2 = find_nearest(wl_emitted, 0.682)# 0.66025   
    for i in range(11, 40):#np.shape(data)[1]):
        for j in range(np.shape(data)[2]):
            if not np.isnan(data[:, i, j]).all():
                guess = (0.5, 0.0003,
                        0.5, 0.1, z)

                bounds = [[0, 0, 0.5, 0, 1], 
          [10, 0.1, 1.5, 5, 5.7618]]
                try:

                    pixel = Pixel(i, j)
                    
                    pixel.average_values()
                    pixel.fit_pixel(guess, bounds, [idx1, idx2])
                    pixel.plot_spectrum(indxs=[idx1, idx2], show = False)
                except ValueError:
                    pass
                except RuntimeError:
                    pass
                if not os.path.exists(f"pixels/{i}_pixels"):
                    os.mkdir(f"pixels/{i}_pixels")
                plt.savefig(f'pixels/{i}_pixels/pixel_({i}, {j}).png')
                plt.close();
                del pixel

def show_img_pixel(pixel, pixx, pixy):
     for i in range(len(pixel)):
         for j in range(len(pixel[i])):
             if pixel[i][j] == 0:
                 pixel[i][j] = np.nan
     plt.imshow(pixel, origin='lower')
     plt.scatter([pixx], [pixy], color='red', s=10)
     plt.colorbar()
     #plt.savefig('C:/Users/redma/Downloads/SII_pixel')
     plt.show()

def step_plot(pixel, idx1, idx2, unc, popt, same_width=True):
    plt.figure(figsize=(9, 9))
    plt.step(wl_emitted[idx1:idx2], pixel[idx1:idx2], where='pre')
    plt.fill_between(wl_emitted[idx1:idx2], pixel[idx1:idx2] - unc[idx1:idx2], pixel[idx1:idx2] + unc[idx1:idx2], alpha=0.2)
    gauss = gaussian2_same_wid
    if not same_width:
        gauss = gaussian2_diff_wid
    plt.plot(np.linspace(wl_emitted[idx1], wl_emitted[idx2], 10000), gauss(np.linspace(wl_emitted[idx1], wl_emitted[idx2], 10000), *popt), ls='--')
    plt.show()

def large_step_plot(*args):
    pixel, idx1, idx2, unc, popts, pcovs, titles, same_widths = args

    fontsize = 22
    fig, ax = plt.subplots(2, len(popts), figsize=(len(popts) * 7, 7), gridspec_kw={'height_ratios' : [2, 1], 'hspace' : 0.05}, sharex=True)
    pos = [0.67166, 0.6731]
    for i in range(len(popts)):
        ax[0][i].step(wl_emitted[idx1:idx2], pixel[idx1:idx2], where='mid')
        ax[0][i].fill_between(wl_emitted[idx1:idx2], pixel[idx1:idx2] - unc[idx1:idx2], pixel[idx1:idx2] + unc[idx1:idx2], alpha=0.2)
        gauss = gaussian2_same_wid
        if not same_widths[i]:
            gauss = gaussian2_diff_wid
        #ax[0][i].set_ylim(0.2, 0.85)
        ax[0][i].plot(np.linspace(wl_emitted[idx1], wl_emitted[idx2], 10000), gauss(np.linspace(wl_emitted[idx1], wl_emitted[idx2], 10000), *popts[i]), ls='--', color='mediumseagreen', zorder=6)
        ax[0][i].set_title(titles[i], fontsize = fontsize)
        ax[0][i].minorticks_on()
        ax[0][i].axvline(0.671644, linestyle='--', color='gray', alpha=0.6, linewidth=1)
        ax[0][i].axvline(0.673081, linestyle='--', color='gray', alpha = 0.6, linewidth=1)
        ax[0][i].set_xlim(wl_emitted[idx1], wl_emitted[idx2])
        ax[0][i].tick_params(axis='both', labelsize= 16)
        amp6716 = str(round(popts[i][0] - popts[i][-2] * 0.67166 - popts[i][-1], 4))
        amp_err6716 = str(round(np.sqrt(pcovs[i][0][0]), 4))
        width6716 = str(round(popts[i][1] * 10000, 4))
        width_err6716 = str(round(pcovs[i][1][1] * 10000, 4))

        amp6731 = str(round(popts[i][2]- popts[i][-2] * 0.6731 - popts[i][-1], 4))
        amp_err6731 = str(round(np.sqrt(pcovs[i][2][2]), 4))
        if same_widths[i]:
            width6731 = width6716
            width_err6731 = width_err6716
        else:
            width6731 = str(round(popts[i][3] * 10000 , 4))
            width_err6731 = str(round(pcovs[i][3][3] * 10000, 4))

        eq_font = 11
        offset = 0.13
        multiplier = 0.422
        # 6716 params
        sII_6716 = f"\\mathrm{{[SII]_{{6716}}}}"
        amp_eq = f"\\mathrm{{Ampl.}} = {amp6716} \\pm {amp_err6716}"
        width_eq = f"\\mathrm{{Width}} = {width6716} \\pm {width_err6716}"
        fig.text(multiplier * i + offset, 0.825, "$%s$"%sII_6716, fontsize=eq_font+2, ha = 'left', va='center')
        fig.text(multiplier * i + offset, 0.8, "$%s$"%amp_eq, fontsize=eq_font, ha = 'left', va='center')
        fig.text(multiplier * i + offset, 0.775, "$%s$"%width_eq, fontsize=eq_font, ha = 'left', va='center')
        
        offset = 0.35
        multiplier = 0.422
        # 6731 params
        sII_6731 = f"\\mathrm{{[SII]_{{6731}}}}"
        amp_eq = f"\\mathrm{{Ampl.}} = {amp6731} \\pm {amp_err6731}"
        width_eq = f"\\mathrm{{Width}} = {width6731} \\pm {width_err6731}"
        fig.text(multiplier * i + offset, 0.825, "$%s$"%sII_6731, fontsize=eq_font+2, ha = 'left', va='center')
        fig.text(multiplier * i + offset, 0.8, "$%s$"%amp_eq, fontsize=eq_font, ha = 'left', va='center')
        fig.text(multiplier * i + offset, 0.775, "$%s$"%width_eq, fontsize=eq_font, ha = 'left', va='center')
        
        # ratio of lines equation
        ratio = round((float(amp6716) / float(amp6731)) * (float(width6716) / float(width6731)), 4)
        ratio_unc = round(ratio * np.sqrt((float(amp_err6716) / float(amp6716))**2 + (float(amp_err6731) / float(amp6731))**2 + (float(width_err6731) / float(width6731))**2 + (float(width_err6716) / float(width6716))**2), 4)
        ratio_eq = f"\\frac{{\\mathrm{{[SII]_{{6716}}}}}}{{\\mathrm{{[SII]_{{6731}}}}}} = {ratio} \\pm {ratio_unc}"
        fig.text(multiplier * i + offset, 0.42, "$%s$"%ratio_eq, fontsize=eq_font + 4, ha='center', va='center')

        residuals = pixel[idx1:idx2] - gauss(wl_emitted[idx1:idx2], *popts[i])
        ax[1][i].scatter(wl_emitted[idx1:idx2], residuals, color='black', zorder=5)
        #ax[1][i].set_ylim(-0.25, 0.25)
        ax[1][i].axhline(0, alpha=0.4, color='gray')
        ax[1][i].fill_between(wl_emitted[idx1:idx2], residuals - unc[idx1:idx2], residuals + unc[idx1:idx2], alpha=0.2)
        ax[1][i].set_xlim(wl_emitted[idx1], wl_emitted[idx2-1])
        ax[1][i].tick_params(axis='both', labelsize= 16)
        ax[1][i].axvline(0.671644, linestyle='--', color='gray', alpha=0.6, linewidth=1)
        ax[1][i].axvline(0.673081, linestyle='--', color='gray', alpha = 0.6, linewidth=1)
        #tickloc, ticklabel = ax[1][i].xticks()
        #ax[1][i].xticks(tickloc, float(ticklabel) * 10000)
        #plt.setp(ax[1][i], xticks = plt.xticks()[0], xticklabels=plt.xticks()[1] * 10000)
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 10000))
        ax[0][i].xaxis.set_major_formatter(ticks_x)
    
    xlabel = "\\mathrm{Wavelength}~(\\mathrm{\\r{A}})"
    fig.text(0.5, 0.05, '$%s$'%xlabel, fontsize=fontsize, ha='center', va='center')
    ylabel = "\\mathrm{MJy/sr}"
    fig.text(0.07, 0.5, '$%s$'%ylabel, fontsize=fontsize, ha='center', va='center', rotation='vertical')
    #plt.savefig(f'C:/Users/redma/Downloads/SII_photos/SII_ratio_fig_{titles[0]}')
    plt.show()
    
    
    
    # add both peaks instead of 67166, add all the parameters to the side, maybe add better error visualization
    # plot all the pixels, ratios, and determine electron density from that