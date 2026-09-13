# Application of ML & PINNs for Predicting Axial Load Capacity of Concrete-Filled Steel Columns

This repository contains machine learning models, metaheuristic optimizations, and analytical implementations for predicting the axial load capacity ($P_{exp}$) of Concrete-Filled Steel Tube (CFST) columns.

## 📌 Overview
Concrete-Filled Steel Tube (CFST) columns are widely used in modern civil engineering structures due to their excellent structural performance, ductility, and high load-bearing capacity. Accurately predicting their ultimate axial load capacity requires accounting for complex interactions between the steel tube and the concrete core (such as confinement effects).

This project implements data preprocessing, model selection, hyperparameter tuning, and metaheuristic optimization algorithms (FPA, SMA, SOS) across multiple ML paradigms:
- **K-Nearest Neighbors (KNN)** (Multiple metric & weight variants)
- **Support Vector Regression (SVR)** ($\nu$-SVR with RBF kernels, Fuzzy SVR)
- **Ensemble & Tree-based Models** (Random Forest, Extra Trees, XGBoost, CatBoost)
- **Bio-Inspired Metaheuristic Algorithms** (Flower Pollination Algorithm, Slime Mould Algorithm, Symbiotic Organisms Search)

## 📁 Repository Structure
```
.
├── CIRC_model_ready.csv    # Prepared dataset of CFST column dimensions, material properties & axial capacities
├── project.ipynb           # Comprehensive Jupyter Notebook with data processing, training, tuning & evaluation
├── ML basics - report.docx # Detailed project documentation report
├── .gitignore              # Git ignore configuration
└── README.md               # Project documentation
```

## 📊 Dataset Parameters
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

## 🚀 Key Results & Best Models
- **CatBoost Regressor (L2 Loss)** & **XGBoost Regressor**: Achieved superior $R^2$ scores and minimum RMSE / MAE error metrics on unseen test data.
- **Metaheuristic Optimization**: Mealpy optimization framework was integrated to fine-tune model parameters dynamically using FPA, SMA, and SOS algorithms.

## 💻 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed along with the following packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost mealpy
```

### Running the Notebook
Launch Jupyter Notebook to explore the code:
```bash
jupyter notebook project.ipynb
```