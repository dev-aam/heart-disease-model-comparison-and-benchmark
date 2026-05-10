# Heart Disease ML Comparison & Benchmark 🫀

A rigorous, end-to-end comparison of **8 machine learning classifiers** for heart disease prediction. Built on merged UCI Cleveland + Kaggle datasets, this project goes beyond simple accuracy reporting — covering nested cross-validation, statistical significance testing, SHAP explainability, and gender fairness analysis.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Methodology](#methodology)
- [Evaluation](#evaluation)
- [Statistical Testing](#statistical-testing)
- [Explainability (SHAP)](#explainability-shap)
- [Fairness Analysis](#fairness-analysis)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)

---

## Overview

Heart disease remains one of the leading causes of mortality worldwide. Early and accurate prediction is critical. This project benchmarks multiple ML classifiers on a clinical dataset to identify which model best balances recall (minimising missed diagnoses), precision, and overall accuracy — while also interrogating fairness and interpretability.

---

## Dataset

Two publicly available datasets are merged into a single clean pipeline:

| Source | Description |
|--------|-------------|
| **UCI Cleveland** (`processed.cleveland.data`) | Classic cardiology dataset with `?` missing values handled via median/mode imputation |
| **Kaggle UCI-derived** (`heart.csv`) | Pre-cleaned variant of the UCI dataset |

- Duplicate rows are removed after merging
- Target variable is binarised: `0` = No Disease, `1` = Disease present
- Features include: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, slope, number of vessels, and thalassemia type

---

## Models

Eight classifiers are compared with full hyperparameter tuning:

| # | Model |
|---|-------|
| 1 | Logistic Regression |
| 2 | Decision Tree |
| 3 | Random Forest |
| 4 | K-Nearest Neighbors (KNN) |
| 5 | Support Vector Machine (SVM) |
| 6 | Gradient Boosting |
| 7 | XGBoost |
| 8 | LightGBM |

---

## Methodology

### Preprocessing
- `StandardScaler` applied to all numeric and binary features (critical for distance-based models like KNN and SVM)
- `OneHotEncoder` (drop first) applied to categorical features
- Built as a `ColumnTransformer` inside a `Pipeline` for leak-free processing

### Nested Cross-Validation

| Loop | Role | Details |
|------|------|---------|
| **Inner** | Hyperparameter selection | `GridSearchCV`, 3-fold Stratified KFold, optimised for Recall |
| **Outer** | Unbiased performance estimation | 5-fold Stratified KFold, fresh `clone` each fold |

- Each outer fold receives a fresh, unfitted clone of the `GridSearchCV` — ensuring reported CV scores are genuine out-of-sample estimates
- Final models are refit on the full training set using the best hyperparameters found

---

## Evaluation

Test set performance is reported with **point estimates and 95% Bootstrap Confidence Intervals** (n = 1,000 resamples) across:

- **Accuracy**
- **Precision**
- **Recall** *(primary metric — clinically, missing a diseased patient is costly)*
- **F1 Score**
- **ROC-AUC**
- **PR-AUC**

Additional visualisations:
- Confusion matrices for all 8 models
- Overlaid ROC curves
- Overlaid Precision-Recall curves
- Overfitting check (train vs. test recall per model)

---

## Statistical Testing

### McNemar's Test
Pairwise McNemar's tests are run across all model combinations to assess whether performance differences are **statistically significant** — not just numerical noise on the test set.

---

## Explainability (SHAP)

SHAP (SHapley Additive exPlanations) values are computed to:
- Identify the most important features globally
- Understand how each feature pushes predictions toward or away from a positive diagnosis
- Provide a model-agnostic, theoretically grounded explanation framework

---

## Fairness Analysis

All 8 models are evaluated separately on **male and female patient subgroups**, reporting:

| Metric | Description |
|--------|-------------|
| Recall Gap | Equal Opportunity — does the model miss diagnoses more for one gender? |
| Precision Gap | Does the model over-predict disease more for one gender? |
| FPR Gap | False positive rate disparity across genders |
| Avg Gap | Overall fairness score (lower is fairer) |

Models are ranked by `Avg Gap` to identify the most equitable classifier.

---

## Tech Stack

```
Python · scikit-learn · XGBoost · LightGBM · SHAP
pandas · NumPy · matplotlib · seaborn · scipy
```

---

## Project Structure

```
heart-disease-ml-benchmark/
│
├── heart_disease_fixed_3_.ipynb   # Main notebook — full pipeline
├── heart.csv                      # Kaggle UCI-derived dataset
├── processed.cleveland.data       # Raw UCI Cleveland dataset
└── README.md
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/heart-disease-ml-benchmark.git
cd heart-disease-ml-benchmark
```

### 2. Install dependencies
```bash
pip install scikit-learn xgboost lightgbm shap pandas numpy matplotlib seaborn scipy
```

### 3. Add the datasets
Place `heart.csv` and `processed.cleveland.data` in the root directory.
- [UCI Cleveland Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

### 4. Run the notebook
```bash
jupyter notebook heart_disease_fixed_3_.ipynb
```

---

## License

This project is open-source and available under the [MIT License](LICENSE).
