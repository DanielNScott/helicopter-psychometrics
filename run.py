# Shared libraries, environment variables, etc
from configs import *

# Project imports
from changepoint.dataio import *
from changepoint.subjects import *
from changepoint.tasks import *
from analysis.aggregation import *
from analysis.analysis import *
from analysis.reliability import *
from estimation.mle import *
from estimation.fim import *
from estimation.main import recovery_analysis
from plotting.figures import *

# Settings
create_figures = [5] #[1,2,3,4,5]
run_recovery = True

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

# Run the parameter recovery analysis
recovery, err_analysis, fim_df = recovery_analysis()

# Figure 1
if 1 in create_figures:
    figure_1(subjs, tasks, subj_pcp_lr, subj_pca_lr_scores)
    compile_figure_1()

# Figure 2
if 2 in create_figures:
    figure_2(tasks, subj_pcp_lr, group_pca_basis, subj_pca_lr_scores, subj_linear_models, savefig=True)
    compile_figure_2()

# Figure 3
if 3 in create_figures:
    figure_3(subjs, tasks, subj_linear_models, subj_pcp_lr, group_pca_basis, subj_pca_lr_scores, group_pca_ve, savefig=True, close=True)
    compile_figure_3()

# Figure 4
if 4 in create_figures:
    figure_4(reliabilities)
    compile_figure_4()

# Figure 5
if 5 in create_figures:
    figure_5(err_analysis, fim_df, recovery['analysis'])
    compile_figure_5()
