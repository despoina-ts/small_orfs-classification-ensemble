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
    y_pred: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """
    Compute sensitivity (recall for the positive class), specificity, and the confusion matrix.

    Sensitivity is computed as recall with pos_label=1.
    Specificity is computed as TN / (TN + FP).

    Parameters
    ----------
    y_true : pandas.Series
        Ground-truth labels.
    y_pred : numpy.ndarray
        Predicted labels.

    Returns
    -------
    tuple of (float, float, numpy.ndarray)
        sensitivity : float
            Recall for the positive class (label = 1).
        specificity : float
            True negative rate for the negative class (label = 0).
        cm : numpy.ndarray
            Confusion matrix with shape (2, 2).
    """
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    specificity = cm[0][0] / (cm[0][0] + cm[0][1])
    return sensitivity, specificity, cm


def evaluate(name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    """
    Print standard classification metrics for a model.

    This function prints:
    - accuracy
    - confusion matrix
    - full classification report (precision/recall/f1 per class)
    - sensitivity (recall for label=1)
    - specificity (true negative rate for label=0)

    Parameters
    ----------
    name : str
        A human-readable model name used in printed output.
    y_true : pandas.Series
        Ground-truth labels.
    y_pred : numpy.ndarray
        Predicted labels.

    Returns
    -------
    None
        This function prints metrics to stdout and does not return a value.
    """
    acc = accuracy_score(y_true, y_pred)
    sens, spec, cm = sensitivity_specificity(y_true, y_pred)
    report = classification_report(y_true, y_pred)

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

    Workflow
    --------
    1) Load `merged_data.csv`
    2) Split features/labels using the column `TRUE_LABELS`
    3) One-hot encode all feature columns
    4) Train Decision Tree, KNN, and Random Forest models
    5) Evaluate each model on the test set
    6) Print label distribution for train/test splits

    Returns
    -------
    None
        This function orchestrates the workflow and prints results to stdout.
    """
    df = pd.read_csv("merged_data.csv")

    y: pd.Series = df["TRUE_LABELS"]
    X: pd.DataFrame = df.drop(columns=["TRUE_LABELS"])

    # One-hot encode all feature columns (categorical + any non-numeric features)
    X = pd.get_dummies(X, columns=X.columns)

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

    # Print label distribution for sanity-checking the split
    train_label0 = (y_train == 0).sum()
    train_label1 = (y_train == 1).sum()
    test_label0 = (y_test == 0).sum()
    test_label1 = (y_test == 1).sum()

    print(f"Training set: {train_label0} label-0, {train_label1} label-1")
    print(f"Testing set:  {test_label0} label-0, {test_label1} label-1")


if __name__ == "__main__":
    main()
