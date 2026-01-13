import pandas as pd


def preprocess_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize classification labels to a consistent set of values.

    The function standardizes common variants of the labels (e.g. "Coding",
    "Non-coding") into:
    - "coding"
    - "noncoding"

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe that must contain a `classification` column.

    Returns
    -------
    pandas.DataFrame
        The same dataframe with normalized values in the `classification` column.
    """
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
    """
    Create a binary label column from the `classification` column.

    Steps performed:
    - Fill missing `classification` values using the most frequent class.
    - Map "coding" -> 0 and "noncoding" -> 1.
    - Drop duplicate rows based on (`SorfID`, `classification`).
    - Print the final binary label counts for sanity checking.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe that must contain `SorfID` and `classification` columns.

    Returns
    -------
    pandas.DataFrame
        Dataframe with an additional `binary_classification` column.
    """
    # Fill missing classification with the most frequent class
    df["classification"] = df["classification"].fillna(
        df["classification"].value_counts().index[0]
    )

    # Map labels to binary values
    mapping = {"coding": 0, "noncoding": 1}
    df["binary_classification"] = df["classification"].map(mapping)

    # Drop duplicates (same SorfID & classification)
    df = df.drop_duplicates(subset=["SorfID", "classification"], keep="first")

    # Print class distribution
    counts = df["binary_classification"].value_counts(dropna=False).to_dict()
    print("Binary counts:", counts)

    return df


def main() -> None:
    """
    Merge multiple tool output CSVs with ground-truth labels and export a clean dataset.

    Workflow
    --------
    1) Load and concatenate multiple tool output files.
    2) Normalize label naming and create a binary label column.
    3) Load ground-truth labels from `TRUE_LABELS.csv`.
    4) Merge on `SorfID`.
    5) Clean selected columns (Probability, SorfAnnotation) if present.
    6) Save the final merged dataset to `merged_data.csv`.

    Returns
    -------
    None
        Writes `merged_data.csv` to disk and prints basic summary information.
    """
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

    # Load and concatenate all tool outputs
    dfs = [pd.read_csv(fn, sep=";") for fn in filenames]
    final_df = pd.concat(dfs, ignore_index=True)

    final_df = preprocess_classification(final_df)
    final_df = add_binary_classification(final_df)

    # Load ground-truth labels
    true_labels_df = pd.read_csv("TRUE_LABELS.csv", sep=";")

    # Merge on SorfID
    merged = pd.merge(final_df, true_labels_df, on="SorfID", how="inner")

    # Ensure the ground-truth label column is named consistently
    if "classification_y" in merged.columns:
        merged = merged.rename(columns={"classification_y": "TRUE_LABELS"})
    elif "classification" in true_labels_df.columns:
        merged = merged.rename(columns={"classification": "TRUE_LABELS"})

    # Remove duplicate classification column from the merge if present
    if "classification_x" in merged.columns:
        merged = merged.drop(columns=["classification_x"])

    # Clean probability column (if present)
    if "Probability" in merged.columns:
        merged["Probability"] = pd.to_numeric(merged["Probability"], errors="coerce")
        merged["Probability"] = merged["Probability"].fillna(merged["Probability"].mean())

    # Clean annotation column (if present)
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
