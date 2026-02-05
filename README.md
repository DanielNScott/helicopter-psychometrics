# Helicopter Psychometrics

Analyzes human learning behavior in a helicopter task paradigm.
- Fits computational models of belief updating and changepoint detection
- Estimates subject-level parameters via regression and PCA
- Computes peri-changepoint learning rate statistics
- Assesses reliability via split-half analysis
- Validates parameter recovery via MLE and Fisher information
- Generates publication figures with SVG composition

## Module Structure

Top-level files:
- `configs.py`: Centralized configuration and shared imports
- `run.py`: Main analysis pipeline
- `supplement.py`: Systematic comparison of linear models on recovery, variance explained, and reliability; generates figure 8
- `alternative_models.py`: Simulates non-Bayesian cognitive models (Rescorla-Wagner, Pearce-Hall, Leaky Integrator) and analyzes them through the Bayesian linear model pipeline
- `diagnostic_betas.py`: Diagnostic script verifying that beta parameters produce distinct subject behavior

Packages:
- `changepoint/`: Task generation and subject modeling
  - `dataio.py`: Experiment data loading from .mat/.csv files
  - `subjects.py`: Subject belief model and simulation
  - `tasks.py`: Changepoint task generation
  - `multiproc.py`: Parallel task generation and subject simulation via multiprocessing
- `analysis/`: Statistical analysis
  - `aggregation.py`: Trial alignment and data subsetting
  - `analysis.py`: Model fitting (linear models, PCA)
  - `reliability.py`: Split-half reliability analysis
- `estimation/`: Parameter estimation
  - `mle.py`: Maximum likelihood estimation and parameter recovery
  - `fim.py`: Fisher information matrix analysis
  - `main.py`: Recovery analysis entry point
- `plotting/`: Visualization
  - `plots_basic.py`: Core plotting functions
  - `plots_fim.py`: Fisher information visualizations
  - `plots_recovery.py`: Parameter recovery plots
  - `figures.py`: Multi-panel figure generation
  - `svgtools.py`: SVG manipulation for figure composition

## Main Execution

The project can be run using run.py.

Doing so will:
1. Read experimental subject responses and task parameters
2. Infer normative CPP and relative uncertainty for each subject
3. Fit linear models of updating (default: `model-pe-cpp-ru-prod-deltas`)
4. Compute learning rates at trials surrounding changepoints
5. Extract principal components from peri-CP learning rates
6. Compute split-half correlations for regression betas and PCA scores
7. Run parameter recovery analysis (OLS-based, with caching)
8. Generate selected figures (controlled by `create_figures` list)

The supplement script (`supplement.py`) runs a systematic model comparison across four linear model variants, computing recovery, variance explained, and reliability for each, and generates figure 8.

## Data

Input data lives in `data/`. The primary input file is `McGuireNassar2014data.csv`, containing experimental subject responses and task parameters. Each row is one trial for one subject, with columns for subject ID, block, trial, observation, prediction, and task parameters (hazard rate, noise SD).

Additional precomputed data files:
- `McGuireNassar2014_betas.csv`: Previously fitted regression betas per subject
- `McGuireNassar2014_peri_cp_lr.csv`: Previously computed peri-changepoint learning rates
- `fitted_parameters.csv`: Previously fitted subject parameters
- `est_corrs_lp_up_loc_unc.csv`: Estimation correlation results

Output directories:
- `data/figures/`: Generated SVG and PDF figures

Configuration paths are defined in `configs.py`: `SUBJ_DATA_DIR`, `SUBJ_DATA_FILE`, `FIGURES_DIR`.

## Subpackages

### Changepoint package

Task generation and subject modeling for the changepoint paradigm.

Modules:
- `dataio.py`: Reads experimental data from CSV/MAT files into standardized format
- `subjects.py`: Subject class with responses, beliefs, and inference/simulation methods
- `tasks.py`: ChangepointTask class and simulation function

The Subject class stores responses and beliefs:
- predictions - subject location estimates per trial
- prediction errors - observation minus prediction
- updates - change in prediction after observation
- learning rates - update divided by prediction error
- CPP - changepoint probability per trial
- relative uncertainty - state uncertainty relative to noise
- model-predicted LRs - normative learning rates from beliefs

Belief inference uses subject prediction errors to compute trial-by-trial CPP and relative uncertainty according to the normative model. The learning rate formula combines these:

```
lr = beta_pe + beta_cpp * cpp + beta_ru * ru * (1 - cpp)
```

Task simulation generates observation sequences with stochastic changepoints, where the latent state jumps to a uniformly random location with hazard rate H.

### Analysis package

Statistical analysis of subject behavior.

Modules:
- `aggregation.py`: Trial alignment relative to changepoints, data subsetting
- `analysis.py`: Linear model fitting, PCA on peri-CP learning rates
- `reliability.py`: Split-half reliability for regression betas and PCA scores

Peri-CP statistics compute learning rates at positions surrounding each changepoint (CP-1, CP0, CP+1, etc.) by regressing update on PE within each position. This yields a profile showing how subjects adjust their learning rate in response to changepoints.

Linear models regress update on PE weighted by CPP and RU. Models are specified by string name and expanded to parameter lists via `get_model_terms` and `terms_to_params`. The available models include:
- `model-pe-cpp-ru`: update ~ pe + pe*cpp + pe*ru
- `model-pe-cpp-ru-prod`: adds pe*cpp*ru interaction term
- `model-pe-cpp-ru-deltas`: adds delta terms (trial-to-trial changes in cpp, ru)
- `model-pe-cpp-ru-prod-deltas`: full model with interactions and deltas

Design matrices are constructed by `build_design_matrix` from PE, CPP, and RU arrays according to the model specification.

PCA on peri-CP learning rates extracts principal components capturing individual differences in changepoint adaptation patterns. Cumulative variance explained by PCA components and linear model terms are compared via `get_lm_cumulative_ve`.

### Estimation package

Parameter estimation and recovery validation.

Modules:
- `mle.py`: Parameter fitting (MLE and OLS) with multiprocessing, parameter recovery studies
- `fim.py`: Fisher information matrix analysis for OLS regression
- `main.py`: Recovery analysis entry points (`recovery_analysis` and `recovery_analysis_high_pe`)

Parameter recovery simulates subjects with known parameters, fits them via MLE or OLS (controlled by `fit_method`), and computes recovery statistics:
- correlation - agreement between true and recovered values
- variance from optimizer - fitting noise across random starts
- variance from behavior - stochasticity in simulated responses
- variance from task - effects of specific task realizations

Fisher information analysis computes the information matrix from regressor covariance, revealing how task properties affect parameter identifiability. Condition number indicates how well beta_cpp and beta_ru can be separately estimated.

### Plotting package

Visualization functions organized by analysis type.

Modules:
- `plots_basic.py`: Task examples, PE-update relationships, peri-CP bars, model comparisons
- `plots_fim.py`: Fisher information contours, task feature correlations
- `plots_recovery.py`: Parameter recovery scatter plots, error covariance analysis
- `figures.py`: Multi-panel figure composition calling plot functions
- `svgtools.py`: SVG scaling, combination, text annotation, PDF export

## Figures

Figures are generated in stages, with plots being saved, then SVG tools compiling the panels into manuscript figures.

The figures are:
1. Task structure, example adaptive vs non-adaptive subjects
2. Peri-CP learning rates, PCA basis, linear model betas
3. Variance explained by PCA vs linear models
4. Split-half reliability of betas and PCA scores
5. Parameter recovery and Fisher information analysis
6. Detailed parameter recovery results
7. High beta_pe recovery analysis (beta_pe in [0.8, 0.98])
8. Cross-model comparison of recovery, reliability, and variance explained (generated by `supplement.py`)
