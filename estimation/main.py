from configs import *
from estimation.mle import (
    parameter_recovery,
    save_recovery,
    load_recovery,
)
from estimation.fim import (
    analyze_error_covariance,
    analyze_task_information,
)

# Parameter recovery analysis 
def recovery_analysis(use_cache=USE_CACHE):

    # Check for results to load if cache flag
    if use_cache:
        try:
            results = load_recovery('n0_recovery')
            return results
        except FileNotFoundError:
            print("No cached recovery results found; running new recovery.")
            use_cache = False

    # No results or cache off, compute recovery params
    if not use_cache:
        blocks = [
            {'ntrials': 120, 'noise_sd': 10},
            {'ntrials': 120, 'noise_sd': 25},
            {'ntrials': 120, 'noise_sd': 10},
            {'ntrials': 120, 'noise_sd': 25},
        ]

        results = parameter_recovery(
            param_names=['beta_cpp', 'beta_ru'],
            n_subjects=100,
            n_tasks_per_subject=5,
            n_reps=5,
            n_refits=1,
            blocks=blocks,
            fit_method='ols',
        )
        save_recovery(results, 'n0_recovery')

    # Analyze recovery results
    err_analysis = analyze_error_covariance(results['true_params'], results['results'])
    fim_df       = analyze_task_information(results['subjects']    , results['tasks'])
    
    return results, err_analysis, fim_df