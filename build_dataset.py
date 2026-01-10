import numpy as np
import pandas as pd


def preprocess_classification(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize classification labels
    df["classification"] = df["classification"].replace(
        {
            "Coding": "coding",
            "coding": "coding",
            "Noncoding": "noncoding",
            "noncoding": "noncoding",
            "Non-coding": "noncoding",
            "non-coding": "noncoding",
        }
    )
    return df


def add_binary_classification(df: pd.DataFrame) -> pd.DataFrame:
    # Fill missing classification with most frequent value
    df["classification"] = df["classification"].fillna(df["classification"].value_counts().index[0])

    # Map labels to binary values
    mapping = {"coding": 0, "noncoding": 1}
    df["binary_classification"] = df["classification"].map(mapping)

    # Drop duplicates (same SorfID & classification)
    df = df.drop_duplicates(subset=["SorfID", "classification"], keep="first")

    # Print counts
    counts = df["binary_classification"].value_counts(dropna=False).to_dict()
    print("Binary counts:", counts)

    return df


def main() -> None:
    filenames = [
        "modified_RF_ensembl.csv",
        "modified_csORF_finder_intronic.csv",
        "modified_csorfinder_ENSEMBLE.csv",
        "modified_deepcpp_ensemble.csv",
        "modified_deepcpp_intronic.csv",
        "modified_ensemble_samba.csv",
        "modified_intronics_samba.csv",
        "modified_RF_intronics.csv",
    ]

    # Load and concat all tool outputs
    dfs = [pd.read_csv(fn, sep=";") for fn in filenames]
    final_df = pd.concat(dfs, ignore_index=True)

    final_df = preprocess_classification(final_df)
    final_df = add_binary_classification(final_df)

    # Load true labels
    true_labels_df = pd.read_csv("TRUE_LABELS.csv", sep=";")

    # Merge on SorfID
    merged = pd.merge(final_df, true_labels_df, on="SorfID", how="inner")

    # Keep TRUE labels with a clean name
    if "classification_y" in merged.columns:
        merged = merged.rename(columns={"classification_y": "TRUE_LABELS"})
    elif "classification" in true_labels_df.columns:
        merged = merged.rename(columns={"classification": "TRUE_LABELS"})

    # Remove duplicated classification column from merge if present
    if "classification_x" in merged.columns:
        merged = merged.drop(columns=["classification_x"])

    # Probability column cleanup
    if "Probability" in merged.columns:
        merged["Probability"] = pd.to_numeric(merged["Probability"], errors="coerce")
        merged["Probability"] = merged["Probability"].fillna(merged["Probability"].mean())

    # SorfAnnotation cleanup
    if "SorfAnnotation" in merged.columns:
        mode_val = merged["SorfAnnotation"].mode()
        fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
        merged["SorfAnnotation"] = merged["SorfAnnotation"].fillna(fill_val)

    merged.to_csv("merged_data.csv", index=False)
    print("Saved: merged_data.csv")
    print("Columns:", list(merged.columns))
    print("Rows:", len(merged))


if __name__ == "__main__":
    main()
