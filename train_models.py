import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


np.random.seed(0)


def sensitivity_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    specificity = cm[0][0] / (cm[0][0] + cm[0][1])
    return sensitivity, specificity, cm


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    sens, spec, cm = sensitivity_specificity(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print(f"{name} نتائج / Results")
    print(f"Accuracy: {acc:.3f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{report}")
    print(f"Sensitivity (Recall): {sens:.3f}")
    print(f"Specificity: {spec:.3f}")
    print("-" * 60)


def main():
    df = pd.read_csv("merged_data.csv")

    y = df["TRUE_LABELS"]
    X = df.drop(columns=["TRUE_LABELS"])

    # One-hot encoding for all columns
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

    train_pos = (y_train == 0).sum()
    train_neg = (y_train == 1).sum()
    test_pos = (y_test == 0).sum()
    test_neg = (y_test == 1).sum()

    print(f"Training set: {train_pos} label-0, {train_neg} label-1")
    print(f"Testing set:  {test_pos} label-0, {test_neg} label-1")


if __name__ == "__main__":
    main()
