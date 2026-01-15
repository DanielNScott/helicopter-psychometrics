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

# Fit linear models to subjects
subj_linear_models = fit_linear_models(subjs, model='m4')

# Get peri-cp statistics
subj_pcp_lr, subj_pcp_cpp, subj_pcp_ru = get_peri_cp_stats(subjs, tasks)

# Fit PCA to peri-CP learning rates
group_pca_basis, subj_pca_lr_scores, group_pca_ve = fit_peri_cp_pca(subj_pcp_lr)