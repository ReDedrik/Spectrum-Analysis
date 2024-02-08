import shutil
import numpy as np
import os
from main import *

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
                pixel_comparison(data[:, i, j], avg_3x3(i, j), xloc=i, yloc=j)
                if not os.path.exists(f"pixels/{i}_pixels"):
                    os.mkdir(f"pixels/{i}_pixels")
                plt.savefig(f'pixels/{i}_pixels/pixel_({i}, {j}).png')
                plt.close();

def h_alpha(pixel, loc=('-', '-')):
        pass