# Alternative (non-Bayesian) cognitive models for the changepoint task.
from configs import *
from changepoint.subjects import Subject, Responses, get_beliefs

# Alternative model parameter ranges
ALT_PARAM_RANGES = {
    'ph': {'eta': (0.05, 0.95), 'alpha_0': (0.05, 0.95)},
    'li': {'lam': (0.05, 0.95), 'gamma': (0.0, 0.3)},
}

# Task blocks
ALT_BLOCKS = [
    {'ntrials': 120, 'noise_sd': 10},
    {'ntrials': 120, 'noise_sd': 25},
    {'ntrials': 120, 'noise_sd': 10},
    {'ntrials': 120, 'noise_sd': 25},
]

N_ALT_SUBJECTS = 200
INIT_PRED = 150.0
PH_SCALE = 150.0
NOISE_SD_UPDATE = 0.5
PRIOR_MEAN = 150.0


def simulate_ph(obs, eta, alpha_0, init_pred=INIT_PRED, noise_sd_update=NOISE_SD_UPDATE,
                scale=PH_SCALE):
    """Pearce-Hall: attention-modulated learning rate that tracks surprise."""
    n = len(obs)
    responses = Responses(n)
    responses.pred[0] = init_pred
    alpha = alpha_0

    for t in range(n):
        responses.pe[t] = obs[t] - responses.pred[t]
        noise = np.random.normal(0, noise_sd_update)
        responses.update[t] = alpha * responses.pe[t] + noise
        responses.pred[t + 1] = np.clip(responses.pred[t] + responses.update[t], 0, 300)
        responses.lr[t] = alpha

        # Attention weight tracks recent surprise magnitude
        alpha = eta * np.abs(responses.pe[t]) / scale + (1 - eta) * alpha
        alpha = np.clip(alpha, 0.0, 1.0)

    return Subject(responses=responses)


def simulate_li(obs, lam, gamma, init_pred=INIT_PRED, noise_sd_update=NOISE_SD_UPDATE,
                prior_mean=PRIOR_MEAN):
    """Leaky integrator with prior pull: exponential decay plus mean reversion."""
    n = len(obs)
    responses = Responses(n)
    responses.pred[0] = init_pred

    for t in range(n):
        responses.pe[t] = obs[t] - responses.pred[t]
        prior_pull = gamma * (prior_mean - responses.pred[t])
        noise = np.random.normal(0, noise_sd_update)
        responses.update[t] = lam * responses.pe[t] + prior_pull + noise
        responses.pred[t + 1] = np.clip(responses.pred[t] + responses.update[t], 0, 300)

        # Effective learning rate
        if responses.pe[t] != 0:
            responses.lr[t] = responses.update[t] / responses.pe[t]
        else:
            responses.lr[t] = lam

    return Subject(responses=responses)


# Dispatch table for simulation functions
SIMULATE_FNS = {
    'ph': simulate_ph,
    'li': simulate_li,
}


def simulate_alt_subjects(model_name, tasks):
    """Simulate many subjects using an alternative model with varied parameters.

    Parameters:
        model_name (str) - 'ph' or 'li'
        tasks (list)     - List of ChangepointTask objects, one per subject

    Returns:
        subjects (list)     - List of Subject instances with .responses populated
        true_params (dict)  - {param_name: array of true values}
    """
    param_ranges = ALT_PARAM_RANGES[model_name]
    sim_fn = SIMULATE_FNS[model_name]
    n = len(tasks)

    # Sample parameters uniformly for each subject
    true_params = {name: np.random.uniform(lo, hi, n) for name, (lo, hi) in param_ranges.items()}

    # Simulate each subject
    subjects = []
    for i in range(n):
        params = {name: true_params[name][i] for name in param_ranges}
        subj = sim_fn(tasks[i].obs, **params)
        subjects.append(subj)

    return subjects, true_params


def infer_beliefs(subjects, tasks):
    """Infer Bayesian beliefs (CPP, RU) from alternative model responses. Modifies subjects in-place."""
    for subj, task in zip(subjects, tasks):
        noise_sd = task.noise_sd if not np.isscalar(task.noise_sd) else np.full(len(task.obs), task.noise_sd)
        subj.beliefs = get_beliefs(subj.responses, task.obs, task.new_blk, task.hazard, noise_sd)
