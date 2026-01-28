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

Packages:
- `changepoint/`: Task generation and subject modeling
  - `dataio.py`: Experiment data loading from .mat/.csv files
  - `subjects.py`: Subject belief model and simulation
  - `tasks.py`: Changepoint task generation
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
3. Fit linear models of updating
4. Compute learning rates at trials surrounding changepoints
5. Extract principal components from peri-CP learning rates
6. Compute split-half correlations for regression betas and PCA scores
7. Compute MLE-based parameter recovey
8. Analyze the recovery statistics
9. Create figures

## Figures

Figures:
- **Figure 1**: Task structure, example adaptive vs non-adaptive subjects
- **Figure 2**: Peri-CP learning rates, PCA basis, linear model betas
- **Figure 3**: Variance explained by PCA vs linear models
- **Figure 4**: Split-half reliability of betas and PCA scores
- **Figure 5**: Parameter recovery and Fisher information analysis
