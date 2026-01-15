# Shared libraries, environment variables, etc
from configs import *

# Project imports
from dataio import *
from subjects import *
from tasks import *
from utils import *
from aggregation import *
from analysis import *
from plots import *
from recovery import *

# Read experimental data
subjs, tasks = read_experiment(max_subj=MAX_SUBJ_NUM)
