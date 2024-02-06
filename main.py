import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import tkinter as tk
import os
import astropy.units as units
from astropy.stats import SigmaClip
from astropy.visualization import simple_norm
from astropy.convolution import convolve
from astropy.io import fits
from astropy.stats import sigma_clipped_stats


file = fits.open("C:/Users/redma/Downloads/SPT2147-50-sigmaclipped-g395m-s3d_v2.fits")