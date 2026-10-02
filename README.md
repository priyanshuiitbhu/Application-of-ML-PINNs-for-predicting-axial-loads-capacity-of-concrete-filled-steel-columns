# Application of ML & PINNs for Predicting Axial Load Capacity of Concrete-Filled Steel Columns

This repository contains machine learning models, metaheuristic optimizations, and analytical implementations for predicting the axial load capacity ($P_{exp}$) of Concrete-Filled Steel Tube (CFST) columns.

## Overview
Concrete-Filled Steel Tube (CFST) columns are widely used in modern civil engineering structures due to their excellent structural performance, ductility, and high load-bearing capacity. Accurately predicting their ultimate axial load capacity requires accounting for complex interactions between the steel tube and the concrete core (such as confinement effects).

This project implements data preprocessing, model selection, hyperparameter tuning, and metaheuristic optimization algorithms (FPA, SMA, SOS) across multiple ML paradigms:
- **K-Nearest Neighbors (KNN)** (Multiple metric & weight variants)
- **Support Vector Regression (SVR)** ($\nu$-SVR with RBF kernels, Fuzzy SVR)
- **Ensemble & Tree-based Models** (Random Forest, Extra Trees, XGBoost, CatBoost)
- **Bio-Inspired Metaheuristic Algorithms** (Flower Pollination Algorithm, Slime Mould Algorithm, Symbiotic Organisms Search)

## Repository Structure
```
.
├── CIRC_model_ready.csv    # Prepared dataset of CFST column dimensions, material properties & axial capacities
├── project.ipynb           # Comprehensive Jupyter Notebook with data processing, training, tuning & evaluation
├── ANN_PINNs.ipynb      # ANN experiments and a planned PINNs section
├── ANN_PINNs_output.json # Saved final ANN cell outputs in Jupyter format
├── ML basics - report.docx # Detailed project documentation report
├── .gitignore              # Git ignore configuration
└── README.md               # Project documentation
```

## Dataset Parameters
- **`D`**: Outer diameter of steel tube (mm)
- **`t`**: Wall thickness of steel tube (mm)
- **`Fy`**: Yield strength of steel (MPa)
- **`fc`**: Compressive strength of concrete core (MPa)
- **`L`**: Column length (mm)
- **`Dt`**: Diameter-to-thickness ratio ($D/t$)
- **`LD`**: Length-to-diameter ratio ($L/D$)
- **`Ac`**: Cross-sectional area of concrete core ($\text{mm}^2$)
- **`As`**: Cross-sectional area of steel tube ($\text{mm}^2$)
- **`SteelCap`**: Steel yield capacity ($A_s \cdot F_y$)
- **`ConcCap`**: Concrete compressive capacity ($A_c \cdot f_c$)
- **`Squash`**: Theoretical squash load ($A_s F_y + A_c f_c$)
- **`Xi`**: Confinement factor ($\xi$)
- **`Pexp`**: Experimental axial load capacity (kN) - *Target Variable*

## Key Results & Best Models
- **CatBoost Regressor (L2 Loss)** & **XGBoost Regressor**: Achieved superior $R^2$ scores and minimum RMSE / MAE error metrics on unseen test data.
- **Metaheuristic Optimization**: Mealpy optimization framework was integrated to fine-tune model parameters dynamically using FPA, SMA, and SOS algorithms.

## ANN Development

The ANN experiments are in [`ANN_PINNs.ipynb`](ANN_PINNs.ipynb). The current
implementation includes:

- Standardization of the input features and target variable.
- A feed-forward network with the architecture `128 -> 128 -> 64 -> 32 -> 1`.
- SiLU activation functions and Xavier weight initialization.
- AdamW optimization with weight decay and a `ReduceLROnPlateau` scheduler.
- Validation-based model selection, gradient clipping, and early stopping.
- Training, validation, and testing metrics including $R^2$, MSE, RMSE, MAE, and MAPE.
- Training-history, actual-versus-predicted, residual, and comparison-table outputs.

The notebook now includes an “Introducing PINNs” heading followed by an empty
code cell. PINNs are planned; no physics-informed model is implemented yet.

The best ANN weights are saved locally as `optimized_ann_model.pt` after training.

The saved optimized run reports a testing R² of **0.9582**, RMSE of **824.1041 kN**,
MAE of **371.2903 kN**, and MAPE of **23.5508%**. It uses an 80/20 train/test
split and reserves 15% of the training partition for validation. These are the
notebook's recorded results; they have not been independently rerun for this update.

The final ANN cell's saved text and chart outputs are also available in
[`ANN_PINNs_output.json`](ANN_PINNs_output.json). PNG outputs use the standard
Jupyter base64 representation.

## Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed along with the following packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost mealpy torch notebook
```

### Running the Notebook
Launch Jupyter Notebook to explore the code:
```bash
jupyter notebook project.ipynb
```

For the ANN notebook and planned PINNs work, run from the repository directory so the
relative dataset path resolves correctly:

```bash
jupyter notebook ANN_PINNs.ipynb
```
