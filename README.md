# ML and PINNs for Axial Load Capacity Prediction of CFST Columns

This repository investigates machine learning and physics-informed neural network
(PINN) methods for predicting the experimental axial load capacity (`Pexp`) of
concrete-filled steel tube (CFST) columns.

CFST capacity depends on nonlinear interactions between the steel tube, concrete
core, geometry, material strength, confinement, and slenderness. The project
compares conventional machine learning models with an artificial neural network
(ANN), then adds interpretability, uncertainty estimation, and a physical
squash-load constraint.

## Project Workflow

The notebooks cover:

- Data loading, validation, visualization, and feature preparation.
- Baseline regression models, including KNN, SVR, tree ensembles, XGBoost, and
  CatBoost.
- Hyperparameter and metaheuristic optimization experiments.
- A PyTorch ANN with standardized inputs and target values.
- SHAP feature-attribution analysis.
- Monte Carlo Dropout uncertainty estimation.
- PINN fine-tuning with a differentiable squash-load constraint.
- Generation of a short methodology report in PDF format.

## Repository Structure

```text
.
├── ANN_PINNs.ipynb          # ANN, SHAP, uncertainty, and PINN experiments
├── ANN_PINNs_output.json    # Saved notebook output representation
├── CIRC_model_ready.csv     # Prepared CFST dataset
├── project.ipynb            # Conventional ML and optimization experiments
├── ML basics - report.docx  # Supporting project report
├── .gitignore
└── README.md
```

Model weights and generated reports are runtime artifacts. The notebooks can
create files such as `optimized_ann_model.pt`, `best_model.pt`, and
`Methodology_Report.pdf` locally.

## Dataset

The target is experimental axial load capacity in kilonewtons.

| Column | Description |
| --- | --- |
| `D` | Outer steel-tube diameter (mm) |
| `t` | Steel-tube wall thickness (mm) |
| `Fy` | Steel yield strength (MPa) |
| `fc` | Concrete compressive strength (MPa) |
| `L` | Column length (mm) |
| `Dt` | Diameter-to-thickness ratio, `D/t` |
| `LD` | Length-to-diameter ratio, `L/D` |
| `Ac` | Concrete-core cross-sectional area (mm²) |
| `As` | Steel-tube cross-sectional area (mm²) |
| `SteelCap` | Steel contribution, `As × Fy` |
| `ConcCap` | Concrete contribution, `Ac × fc` |
| `Squash` | Theoretical squash load |
| `Xi` | Confinement factor |
| `Pexp` | Experimental axial capacity (kN), the prediction target |

## ANN Implementation

[`ANN_PINNs.ipynb`](ANN_PINNs.ipynb) builds a feed-forward PyTorch network with:

- Hidden layers of `128 → 128 → 64 → 32` neurons.
- SiLU activation and Dropout regularization.
- Xavier weight initialization.
- AdamW optimization and a `ReduceLROnPlateau` scheduler.
- Gradient clipping, validation-based checkpointing, and early stopping.
- MSE, RMSE, MAE, MAPE, and R² evaluation metrics.

The notebook automatically selects CUDA, Apple MPS, or CPU. Tensors are moved to
the model's device before training and inference.

Before running the PINN section, execute the setup cell immediately below
`Introduction to PINNs`. It restores the dataset split, scalers, DataLoaders,
model, loss function, and optimizer when the notebook has been opened with a
fresh or partially reset kernel. If the optimized ANN is not already in memory,
the cell loads `optimized_ann_model.pt` from the project directory.

The recorded optimized ANN test results are:

| Metric | Value |
| --- | ---: |
| R² | 0.9582 |
| RMSE | 824.1041 kN |
| MAE | 371.2903 kN |
| MAPE | 23.5508% |

These values are saved notebook results and can vary when the model is retrained.

## Model Interpretation and Uncertainty

SHAP's `GradientExplainer` is used because the ANN contains SiLU activations.
The explanation runs on a CPU copy of the trained model, leaving the original
model unchanged.

Monte Carlo Dropout keeps Dropout layers active during repeated inference. The
mean prediction represents expected capacity, while the standard deviation
estimates epistemic uncertainty. Results are converted from the standardized
target scale back to kN before plotting.

## Physics-Informed Fine-Tuning

The PINN loss combines:

1. Data loss: mean squared error between predicted and experimental capacity.
2. Physics loss: a penalty when predicted capacity exceeds the theoretical
   squash load.

The squash load is calculated from the unscaled physical features:

```text
P_squash = (Ac × fc + As × Fy) / 1000
```

The physics violation is normalized to the target scale before it is combined
with the data loss. This keeps the two loss terms numerically comparable.

## Installation

Python 3.10 or newer is recommended. Create and activate a virtual environment,
then install the notebook dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  jupyter pandas numpy matplotlib seaborn scikit-learn \
  torch xgboost catboost mealpy shap fpdf2
```

`fpdf2` provides the `from fpdf import FPDF` import used by the methodology-report
cell.

## Running the Notebooks

Start Jupyter from the repository directory so the relative dataset path works:

```bash
jupyter notebook
```

Open `project.ipynb` for conventional ML experiments or `ANN_PINNs.ipynb` for the
ANN and PINN workflow. Run `ANN_PINNs.ipynb` from top to bottom when possible.
If the kernel was restarted or only the PINN section is needed, run the setup
cell below `Introduction to PINNs` before the PINN loss and training cells. It
recreates the required preprocessing objects and loads `optimized_ann_model.pt`
when the trained model is not already available.

The PINN training cell uses the initialized `optimizer` and reports total,
data, and physics losses every 50 epochs. The evaluation cells convert the
standardized predictions back to kN before calculating R², RMSE, and MAE.

Training uses random initialization and random train/validation splits, so exact
metrics can differ between runs unless all random seeds are fixed.
