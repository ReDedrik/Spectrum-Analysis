import shutil
import numpy as np
import os
from data_handling import *
import matplotlib.pyplot as plt

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
