# Shared libraries, environment variables, etc
from configs import *

# Project imports
from changepoint.dataio import *
from changepoint.subjects import *
from changepoint.tasks import *
from analysis.aggregation import *
from analysis.analysis import *
from analysis.reliability import *
from analysis.comparison import model_comparison_analysis
from analysis.alternatives import analyze_alternative_models
from estimation.mle import *
from estimation.fim import *
from estimation.main import recovery_analysis
from plotting.figures import *
from plotting.compile import *

# Settings
create_figures = [6] #[1,2,3,4,5]

# Read experimental data
subjs, tasks = read_experiment(max_subj=MAX_SUBJ_NUM)

# Fit linear models to subjects
subj_linear_models = fit_linear_models(subjs, model='model-pe-cpp-ru-prod-deltas')

# Get peri-cp statistics
subj_pcp_lr, subj_pcp_cpp, subj_pcp_ru = get_peri_cp_stats(subjs, tasks)

# Fit PCA to peri-CP learning rates
group_pca_basis, subj_pca_lr_scores, group_pca_ve = fit_peri_cp_pca(subj_pcp_lr)

# Split half reliability analyses (multi-dataset)
reliabilities = multi_dataset_split_half_analysis(nreps=20, verbose=1)

# Run the parameter recovery analysis
recovery, err_analysis, fim_df = recovery_analysis()

# Run model comparison analysis
comparison = model_comparison_analysis(subjs, tasks)

# Run alternative model analysis
alt_analysis = analyze_alternative_models(verbose=1)
real_lm = fit_linear_models(subjs, model=alt_analysis['lm_model'])

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
    figure_4(reliabilities, DATASET_CONFIG)
    compile_figure_4()

# Figure 5: parameter recovery and estimation analysis
if 5 in create_figures:
    figure_5(err_analysis, recovery['analysis'])
    compile_figure_5()

# Figure 6: recovery SD as function of beta_pe
if 6 in create_figures:
    figure_6(recovery)
    compile_figure_6()

# Figure 7: model comparison (recovery, reliability, VE)
if 7 in create_figures:
    figure_7(comparison)
    compile_figure_7()

# Figure 8: alternative cognitive models
if 8 in create_figures:
    figure_8(alt_analysis, real_lm)
    compile_figure_8()

# Figure 9: task information analysis
if 9 in create_figures:
    figure_9(fim_df)
    compile_figure_9()
