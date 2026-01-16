# Shared libraries, environment variables, etc
from configs import *

# Project imports
from dataio import *
from subjects import *
from tasks import *
from aggregation import *
from analysis import *
from figures import *
from recovery import *
from reliability import *

# Read experimental data
subjs, tasks = read_experiment(max_subj=MAX_SUBJ_NUM)

# Fit linear models to subjects
subj_linear_models = fit_linear_models(subjs, model='m4')

# Get peri-cp statistics
subj_pcp_lr, subj_pcp_cpp, subj_pcp_ru = get_peri_cp_stats(subjs, tasks)

# Fit PCA to peri-CP learning rates
group_pca_basis, subj_pca_lr_scores, group_pca_ve = fit_peri_cp_pca(subj_pcp_lr)

# Split half reliability analyses
reliabilities = do_split_half_analysis(subjs, tasks, nreps=20)


# Generate figures
# figure_1(subjs, tasks, subj_pcp_lr, subj_pca_lr_scores)
# compile_figure_1()

# figure_2(tasks, subj_pcp_lr, group_pca_basis, subj_pca_lr_scores, subj_linear_models, savefig=True)
# compile_figure_2()

# figure_3(subjs, tasks, subj_linear_models, subj_pcp_lr, group_pca_basis, subj_pca_lr_scores, group_pca_ve, savefig=True, close=True)
# compile_figure_3()

figure_4(reliabilities)
compile_figure_4()