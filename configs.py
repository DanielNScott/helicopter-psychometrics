# Disable numpy multi-threading for multiprocessing compatibility
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Shared imports
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Interactive plotting
plt.ion()

# Configuration variables
FIT_SUBJECT_PARAMS = False
SUBJ_DATA_FILE = 'McGuireNassar2014data.csv'
SUBJ_DATA_DIR = './data/'
MAX_SUBJ_NUM = 100 # Reduce for debugging