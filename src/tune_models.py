
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]),
        "params": {
            "model__C": [0.1, 1, 10],
            "smote__k_neighbors": [3, 5]
        }
    },
    "Random Forest": {
        "pipeline": Pipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", RandomForestClassifier(
                random_state=42,
                n_jobs=1
            ))
        ]),
        "params": {
            "model__n_estimators": [50],
            "model__max_depth": [10],
            "smote__k_neighbors": [5]
        }
    }
}

tuned_results = []

for name, config in models.items():
    print(f"\nTuning {name}...", flush=True)

    search = GridSearchCV(
        estimator=config["pipeline"],
        param_grid=config["params"],
        scoring="roc_auc",
        cv=2,
        n_jobs=1,
        verbose=2
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)

    print("Best parameters:", search.best_params_)
    print(f"Best cross-validation ROC-AUC: {search.best_score_:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")

    tuned_results.append({
        "Model": name,
        "Best Parameters": str(search.best_params_),
        "CV ROC-AUC": search.best_score_,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test ROC-AUC": roc_auc
    })

results_df = pd.DataFrame(tuned_results)

results_df.to_csv(
    "reports/tuned_model_results.csv",
    index=False
)

print("\nTuned Model Comparison:")
print(results_df.to_string(index=False))
print("\nResults saved to reports/tuned_model_results.csv")
