# Helicopter Psychometrics

Analyzes human learning behavior in a helicopter task paradigm.
- Fits computational models of belief updating and changepoint detection
- Estimates subject-level parameters via regression and PCA
- Computes peri-changepoint learning rate statistics
- Generates publication figures with SVG composition

## Module Structure

Top level files coordinate data loading, analysis, and figure generation.

Files:
- `configs.py`: Centralized configuration and shared imports
- `run.py`: Main analysis pipeline
- `dataio.py`: Experiment data loading from .mat/.csv files
- `subjects.py`: Subject belief model and simulation
- `tasks.py`: Changepoint task generation
- `analysis.py`: Model fitting (linear models, PCA)
- `aggregation.py`: Trial alignment and data subsetting
- `plots.py`: Visualization functions
- `figures.py`: Multi-panel figure generation
- `svgtools.py`: SVG manipulation for figure composition
- `utils.py`: Utility functions

## Data Flow

The analysis pipeline proceeds as:

1. **Load data**: Read experimental subject responses and task parameters
2. **Compute beliefs**: Infer normative CPP and relative uncertainty for each subject
3. **Peri-CP statistics**: Compute learning rates at trials surrounding changepoints
4. **Model fitting**: Fit linear models (update ~ PE * CPP * RU) and PCA on peri-CP LRs
5. **Figure generation**: Create individual panels, compose into multi-panel figures

## Subject Model

Subjects observe noisy samples from a latent state that occasionally changes. The normative model computes:
- **CPP (changepoint probability)**: Likelihood that a changepoint just occurred
- **Relative uncertainty**: Uncertainty about state estimate relative to observation noise
- **Learning rate**: Weight on prediction error for updating beliefs

Subject behavior is characterized by how their learning rates deviate from normative predictions, captured via regression coefficients (beta_PE, beta_CPP, beta_RU) and PCA scores.

## Figures

Figure generation uses a two-stage process:
1. Individual panels saved as SVGs via matplotlib
2. Panels composed horizontally/vertically and labeled using svgtools

Figures:
- **Figure 1**: Task explanation, example adaptive vs non-adaptive subjects
- **Figure 2**: Normative model predictions, LR components, correlation matrix
- **Figure 3**: Model fit quality, beta distributions, variance explained
