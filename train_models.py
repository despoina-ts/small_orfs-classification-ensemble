import numpy as np
import pandas as pd

from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


np.random.seed(0)


def sensitivity_specificity(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """
    Compute sensitivity (recall for the positive class), specificity, and confusion matrix.

    Notes
    -----
    This function assumes binary labels encoded as 0/1 where:
    - 1 is the positive class
    - 0 is the negative class
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    sensitivity = recall_score(y_true, y_pred, pos_label=1)

    tn = cm[0, 0]
    fp = cm[0, 1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return sensitivity, specificity, cm


def evaluate(name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    """
    Print standard classification metrics for a model.
    """
    acc = accuracy_score(y_true, y_pred)
    sens, spec, cm = sensitivity_specificity(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=[0, 1])

    print(f"{name} results")
    print(f"Accuracy: {acc:.3f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{report}")
    print(f"Sensitivity (Recall, label=1): {sens:.3f}")
    print(f"Specificity (TNR, label=0): {spec:.3f}")
    print("-" * 60)


def main() -> None:
    """
    Load the dataset, train multiple classifiers, and evaluate them on a test split.
    """
    df = pd.read_csv("merged_data.csv")

    # Convert TRUE_LABELS into a binary target.
    # This makes the script robust whether TRUE_LABELS is "coding/noncoding" or already 0/1.
    if df["TRUE_LABELS"].dtype == object:
        mapping = {"coding": 0, "noncoding": 1}
        y = df["TRUE_LABELS"].astype(str).str.strip().str.lower().map(mapping)
    else:
        y = df["TRUE_LABELS"]

    # Drop rows with unknown labels (if any)
    valid = y.isin([0, 1])
    df = df.loc[valid].copy()
    y = y.loc[valid].astype(int)

    X = df.drop(columns=["TRUE_LABELS"])

    # One-hot encode non-numeric columns only (safer than encoding everything)
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0, stratify=y
    )

    clf_dt = DecisionTreeClassifier(random_state=0)
    clf_knn = KNeighborsClassifier()
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=0)

    clf_dt.fit(X_train, y_train)
    clf_knn.fit(X_train, y_train)
    clf_rf.fit(X_train, y_train)

    pred_dt = clf_dt.predict(X_test)
    pred_knn = clf_knn.predict(X_test)
    pred_rf = clf_rf.predict(X_test)

    evaluate("Decision Tree (test)", y_test, pred_dt)
    evaluate("KNN (test)", y_test, pred_knn)
    evaluate("Random Forest (test)", y_test, pred_rf)

    train_label0 = int((y_train == 0).sum())
    train_label1 = int((y_train == 1).sum())
    test_label0 = int((y_test == 0).sum())
    test_label1 = int((y_test == 1).sum())

    print(f"Training set: {train_label0} label-0, {train_label1} label-1")
    print(f"Testing set:  {test_label0} label-0, {test_label1} label-1")


if __name__ == "__main__":
    main()
