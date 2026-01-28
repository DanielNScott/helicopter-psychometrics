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
from mle import *
from fim import *

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

# Parameter recovery analysis (for figure 5)
if run_recovery:
    blocks = [
        {'ntrials': 120, 'noise_sd': 10},
        {'ntrials': 120, 'noise_sd': 25},
        {'ntrials': 120, 'noise_sd': 10},
        {'ntrials': 120, 'noise_sd': 25},
    ]

    recovery_result = parameter_recovery(
        param_names=['beta_cpp', 'beta_ru'],
        n_subjects=100,
        n_tasks_per_subject=5,
        n_reps=5,
        n_refits=1,
        blocks=blocks,
        fit_method='ols',
    )
    save_recovery(recovery_result, 'n0_recovery')

else:
    recovery_result = load_recovery('n0_recovery')


# Analyze recovery results
err_analysis = analyze_error_covariance(recovery_result['true_params'], recovery_result['results'])
fim_df = analyze_task_information(recovery_result['subjects'], recovery_result['tasks'])


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
    figure_5(err_analysis, fim_df, recovery_result['analysis'])
    compile_figure_5()
