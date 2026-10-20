
# Fraud Detection Pipeline

A supervised machine learning project that compares Logistic Regression and Random Forest for detecting fraudulent credit card transactions in a highly imbalanced dataset.

## Project Objectives

- Explore and understand an imbalanced transaction dataset.
- Apply SMOTE within machine learning pipelines.
- Train Logistic Regression and Random Forest classifiers.
- Tune hyperparameters using GridSearchCV.
- Evaluate models using Precision, Recall, F1-Score, and ROC-AUC.
- Compare baseline and tuned model performance.

## Dataset

This project uses the Credit Card Fraud Detection dataset, containing 284,807 transactions and 31 columns.

- `Class = 0`: Legitimate transaction
- `Class = 1`: Fraudulent transaction
- Fraudulent transactions: 492
- Fraud rate: approximately 0.173%

**Dataset setup:** Download the Credit Card Fraud Detection dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place `creditcard.csv` inside the project's `data/` directory:

`data/creditcard.csv`

The dataset is excluded from Git tracking through `.gitignore`. Download it separately and follow the dataset's applicable terms.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib and Seaborn

## Project Structure

```text
Fraud-Detection-Pipeline/
|-- data/
|   |-- creditcard.csv  (download separately)
|-- reports/
|   |-- class_distribution.png
|   |-- detailed_evaluation.csv
|   |-- logistic_regression_confusion_matrix.png
|   |-- model_comparison.csv
|   |-- model_performance_dashboard.png
|   |-- random_forest_confusion_matrix.png
|   |-- tuned_model_results.csv
|-- src/
|   |-- check_data.py
|   |-- create_dashboard.py
|   |-- explore_data.py
|   |-- train_models.py
|   |-- evaluate_models.py
|   |-- tune_models.py
|-- .gitignore
|-- README.md
|-- requirements.txt
```

## Methodology

1. Load and inspect the dataset.
2. Explore the distribution of normal and fraudulent transactions.
3. Split the dataset into stratified training and testing sets.
4. Apply SMOTE inside the training pipeline to address class imbalance without resampling the test set.
5. Train Logistic Regression and Random Forest models.
6. Evaluate baseline models using Precision, Recall, F1-Score, and ROC-AUC.
7. Use GridSearchCV with ROC-AUC scoring for hyperparameter selection.
8. Evaluate tuned models on the held-out test set.
9. Generate a combined performance dashboard.

## Baseline Results

| Model | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.0578 | 0.9184 | 0.9708 |
| Random Forest | 0.8351 | 0.8265 | 0.9644 |

## Tuned Model Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Cross-validation ROC-AUC | 0.9754 | 0.9825 |
| Test Precision | 0.0582 | 0.5148 |
| Test Recall | 0.9184 | 0.8878 |
| Test ROC-AUC | 0.9709 | 0.9853 |

## Best Hyperparameters

**Logistic Regression**
- `model__C`: 0.1
- `smote__k_neighbors`: 5

**Random Forest**
- `model__n_estimators`: 50
- `model__max_depth`: 10
- `smote__k_neighbors`: 5

## Key Findings

- Tuned Random Forest achieved a test ROC-AUC of approximately 0.9853.
- Its test Recall was approximately 0.8878, while Precision was approximately 0.5148.
- Logistic Regression achieved higher Recall at approximately 0.9184, but its low Precision indicates many false-positive predictions.
- The results illustrate the trade-off between identifying fraudulent transactions and incorrectly flagging legitimate transactions.
- Accuracy alone can be misleading when the dataset is highly imbalanced.

The tuned Random Forest configuration was evaluated with the selected hyperparameters. Its GridSearchCV parameter grid contained only one Random Forest combination, so broader Random Forest tuning could be explored in future work.

## Dashboard

The combined visualization is saved at:

`reports/model_performance_dashboard.png`

It presents the class distribution, baseline metrics, tuned metrics, and the tuned models' Precision–Recall trade-off.

## How to Run

Run these commands from the project root directory after downloading the dataset.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the project scripts:

```bash
python src/check_data.py
python src/explore_data.py
python src/train_models.py
python src/evaluate_models.py
python src/tune_models.py
python src/create_dashboard.py
```

## Future Improvements

- Explore a broader hyperparameter search for Random Forest.
- Tune classification thresholds.
- Compare alternative resampling strategies.
- Analyze the cost of false positives and missed fraud cases.

## Disclaimer

This project is for educational and research purposes. Performance on this dataset does not guarantee similar results in real-world fraud detection systems.
