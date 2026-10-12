"""Train and compare models for heart disease prediction, then save the best pipeline.

Usage:
    python train.py                       # reads data/heart.csv
    python train.py --csv path/to/heart.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import CATEGORICAL, FEATURES, NUMERIC, TARGET, clean

HERE = Path(__file__).parent
ART = HERE / "artifacts"


def build_pipeline(model) -> Pipeline:
    """Imputation, scaling and encoding live inside the pipeline, so they are fit on
    training data only and applied identically at prediction time."""
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", pre), ("model", model)])


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVM": CalibratedClassifierCV(SVC(), ensemble=False),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
}


def main(csv_path: str | None = None) -> None:
    ART.mkdir(exist_ok=True)
    df = clean(pd.read_csv(csv_path or HERE / "data" / "heart.csv"))
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Compare models with cross-validation on the TRAINING data only.
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for name, model in MODELS.items():
        s = cross_validate(build_pipeline(model), X_train, y_train, cv=cv,
                           scoring=["roc_auc", "accuracy", "recall", "f1"])
        rows.append({"model": name,
                     "cv_roc_auc": s["test_roc_auc"].mean(),
                     "cv_accuracy": s["test_accuracy"].mean(),
                     "cv_recall": s["test_recall"].mean(),
                     "cv_f1": s["test_f1"].mean()})
    table = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)
    print(table.round(4).to_string(index=False))
    best = table.loc[0, "model"]

    # Touch the test set once, with the chosen model only.
    pipe = build_pipeline(MODELS[best]).fit(X_train, y_train)
    pred, proba = pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]
    test = {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(), "n_test": len(y_test)}
    print(f"\nBest model: {best}")
    print(json.dumps(test, indent=2))

    # Deployed model: refit on all data.
    final = build_pipeline(MODELS[best]).fit(X, y)
    joblib.dump(final, ART / "model.joblib")
    meta = {"best_model": best, "test": test, "cv_table": table.to_dict("records"),
            "n_rows": len(df), "positive_rate": float(y.mean()),
            "sklearn_version": sklearn.__version__}
    (ART / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved artifacts to", ART)
    print(f"\nNow commit artifacts/ and set this line in requirements.txt:\n  scikit-learn=={sklearn.__version__}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    main(ap.parse_args().csv)
