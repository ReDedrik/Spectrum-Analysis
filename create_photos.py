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
    for i in range(np.shape(data)[1]):
        for j in range(np.shape(data)[2]):
            if not np.isnan(data[:, i, j]).all():
                #plot_avg_3x3(i, j)
                pixel_comparison(data[:, i, j], avg_5x5(i, j), loc=(i, j))
                if not os.path.exists(f"pixels/{i}_pixels"):
                    os.mkdir(f"pixels/{i}_pixels")
                plt.savefig(f'pixels/{i}_pixels/pixel_({i}, {j}).png')
                plt.close();

def show_img_pixel(pixel, pixx, pixy):
     plt.imshow(pixel, origin='lower')
     plt.scatter([pixx], [pixy], color='red', s=10)
     plt.colorbar()
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
        ax[0][i].axvline(0.67166, linestyle='--', color='gray', alpha=0.6, linewidth=1)
        ax[0][i].axvline(0.6731, linestyle='--', color='gray', alpha = 0.6, linewidth=1)
        ax[0][i].set_xlim(wl_emitted[idx1], wl_emitted[idx2])
        ax[0][i].tick_params(axis='both', labelsize= 16)
        amp6716 = str(round(popts[i][0] - popts[i][-2] * 0.67166 - popts[i][-1], 4))
        amp_err6716 = str(round(np.sqrt(pcovs[i][0][0]), 4))
        width6716 = str(round(popts[i][1], 4))
        width_err6716 = str(round(pcovs[i][1][1], 4))
        
        amp6731 = str(round(popts[i][2]- popts[i][-2] * 0.6731 - popts[i][-1], 4))
        amp_err6731 = str(round(np.sqrt(pcovs[i][2][2]), 4))
        width6731 = str(round(popts[i][3], 4))
        width_err6731 = str(round(pcovs[i][3][3], 4))

        equation = f'{amp6716} \\pm {amp_err6716}~\\mathrm{{exp}}\\left( -\\frac{{(x - 0.67166)^2}}{{({width6716} \\pm {width_err6716})^2}} \\right)'
        fig.text(0.35 * i + 0.35, 0.8, f"m = {round(popts[i][-2], 4)}, C = {round(popts[i][-1], 4)}")
        fig.text(0.35 * i + 0.35, 0.4, '$%s$'%equation, fontsize=16, ha='center', va='center')

        residuals = pixel[idx1:idx2] - gauss(wl_emitted[idx1:idx2], *popts[i])
        ax[1][i].scatter(wl_emitted[idx1:idx2], residuals, color='black', zorder=5)
        #ax[1][i].set_ylim(-0.25, 0.25)
        ax[1][i].axhline(0, alpha=0.4, color='gray')
        ax[1][i].fill_between(wl_emitted[idx1:idx2], residuals - unc[idx1:idx2], residuals + unc[idx1:idx2], alpha=0.2)
        ax[1][i].set_xlim(wl_emitted[idx1], wl_emitted[idx2-1])
        ax[1][i].tick_params(axis='both', labelsize= 16)
        ax[1][i].axvline(0.67166, linestyle='--', color='gray', alpha=0.6, linewidth=1)
        ax[1][i].axvline(0.6731, linestyle='--', color='gray', alpha = 0.6, linewidth=1)

    
    xlabel = "\\mathrm{Wavelength}~(\\mu \\mathrm{m})"
    fig.text(0.5, 0.05, '$%s$'%xlabel, fontsize=fontsize, ha='center', va='center')
    ylabel = "\\mathrm{MJy/sr}"
    fig.text(0.07, 0.5, '$%s$'%ylabel, fontsize=fontsize, ha='center', va='center', rotation='vertical')
    plt.show()

    # add both peaks instead of 67166, add all the parameters to the side, maybe add better error visualization
    # fix sig figs on width