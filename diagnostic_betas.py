"""Diagnostic: verify that beta parameters produce distinct subject behavior."""
from configs import *
from changepoint.subjects import Subject, simulate_subject, DEFAULT_PARAMS_SUBJ
from changepoint.tasks import simulate_cpt

# Generate one shared task
blocks = [
    {'ntrials': 120, 'noise_sd': 10},
    {'ntrials': 120, 'noise_sd': 25},
]
task = simulate_cpt(blocks=blocks)
ntrials = 85

# Define subjects with different beta configurations
beta_configs = [
    {'label': 'Normative (pe=0, cpp=1, ru=1)',     'beta_pe': 0.0, 'beta_cpp': 1.0, 'beta_ru': 1.0},
    {'label': 'High PE (pe=0.5, cpp=0.5, ru=0.5)', 'beta_pe': 0.5, 'beta_cpp': 0.5, 'beta_ru': 0.5},
    {'label': 'PE only (pe=0.4, cpp=0, ru=0)',      'beta_pe': 0.4, 'beta_cpp': 0.0, 'beta_ru': 0.0},
    {'label': 'Low LR (pe=0.1, cpp=0, ru=0)',       'beta_pe': 0.1, 'beta_cpp': 0.0, 'beta_ru': 0.0},
]

# Simulate each subject and collect results
subjects = []
for cfg in beta_configs:
    params = DEFAULT_PARAMS_SUBJ.copy()
    params['noise_sd'] = task.noise_sd[0] if not np.isscalar(task.noise_sd) else task.noise_sd
    params['hazard'] = task.hazard[0] if not np.isscalar(task.hazard) else task.hazard
    params['beta_pe'] = cfg['beta_pe']
    params['beta_cpp'] = cfg['beta_cpp']
    params['beta_ru'] = cfg['beta_ru']
    params['noise_sd_update'] = 0.5

    subj = Subject()
    simulate_subject(subj, task.obs, params)
    subjects.append(subj)

# Plot: 4 rows x 3 columns (task+preds, PE vs update, CPP/RU/LR)
fig, axes = plt.subplots(len(beta_configs), 3, figsize=(16, 3.5 * len(beta_configs)))

for row, (subj, cfg) in enumerate(zip(subjects, beta_configs)):

    # Column 1: task observations and predictions
    ax = axes[row, 0]
    ax.plot(task.obs[:ntrials], '.', alpha=0.5, label='Obs')
    ax.plot(subj.responses.pred[:ntrials], '-', label='Pred')
    ax.set_ylabel('Value')
    ax.set_title(cfg['label'])
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)

    # Column 2: PE vs update scatter
    ax = axes[row, 1]
    ax.scatter(subj.responses.pe, subj.responses.update, alpha=0.3, s=10)
    lims = [-100, 100]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    ax.set_xlabel('PE')
    ax.set_ylabel('Update')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title('PE vs Update')
    ax.grid(alpha=0.3)

    # Column 3: CPP, RU, LR traces
    ax = axes[row, 2]
    ax.plot(subj.beliefs.cpp[:ntrials], label='CPP')
    ax.plot(subj.beliefs.relunc[:ntrials], label='RU')
    ax.plot(subj.responses.lr[:ntrials], 'k--', alpha=0.5, label='LR')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Value')
    ax.set_title('CPP / RU / LR')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

# Label bottom row x-axes
for ax in axes[-1, :]:
    ax.set_xlabel('Trial')

fig.tight_layout()

import os
os.makedirs('./data/figures/', exist_ok=True)
fig.savefig('./data/figures/diagnostic_beta_behavior.svg', dpi=300)
print("Saved to ./data/figures/diagnostic_beta_behavior.svg")

# Print summary stats
print("\nSummary statistics:")
for subj, cfg in zip(subjects, beta_configs):
    lr = subj.responses.lr
    print(f"  {cfg['label']}:")
    print(f"    LR mean={np.nanmean(lr):.3f}, std={np.nanstd(lr):.3f}, min={np.nanmin(lr):.3f}, max={np.nanmax(lr):.3f}")
    print(f"    PE std={np.nanstd(subj.responses.pe):.1f}")
