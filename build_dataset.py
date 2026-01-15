import argparse
from pathlib import Path
import pandas as pd


def preprocess_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize classification labels to a consistent set of values.

    The function standardizes common variants of the labels (e.g. "Coding",
    "Non-coding") into:
    - "coding"
    - "noncoding"
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
    """
    df["classification"] = df["classification"].fillna(
        df["classification"].value_counts().index[0]
    )

    mapping = {"coding": 0, "noncoding": 1}
    df["binary_classification"] = df["classification"].map(mapping)

    df = df.drop_duplicates(subset=["SorfID", "classification"], keep="first")

    counts = df["binary_classification"].value_counts(dropna=False).to_dict()
    print("Binary counts:", counts)

    return df


def main() -> None:
    """
    Merge tool output CSVs with ground-truth labels and export a clean dataset.

    This version is GitHub-safe: it does not use private hardcoded filenames.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tool-dir",
        default="data",
        help="Directory containing tool output CSV files (e.g. sample_tool_outputs.csv)",
    )
    parser.add_argument(
        "--pattern",
        default="sample_tool_outputs.csv",
        help="Filename pattern for tool outputs (default: sample_tool_outputs.csv)",
    )
    parser.add_argument(
        "--true-labels",
        default="data/sample_true_labels.csv",
        help="Path to TRUE_LABELS CSV",
    )
    parser.add_argument(
        "--sep",
        default=";",
        help="CSV separator (default ';')",
    )
    parser.add_argument(
        "--out",
        default="merged_data.csv",
        help="Output CSV filename",
    )
    args = parser.parse_args()

    tool_paths = sorted(Path(args.tool_dir).glob(args.pattern))
    if not tool_paths:
        raise FileNotFoundError(
            f"No tool CSV files found in '{args.tool_dir}' matching '{args.pattern}'"
        )

    dfs = [pd.read_csv(p, sep=args.sep) for p in tool_paths]
    final_df = pd.concat(dfs, ignore_index=True)

    final_df = preprocess_classification(final_df)
    final_df = add_binary_classification(final_df)

    true_labels_df = pd.read_csv(args.true_labels, sep=args.sep)

    merged = pd.merge(final_df, true_labels_df, on="SorfID", how="inner")

    if "classification_y" in merged.columns:
        merged = merged.rename(columns={"classification_y": "TRUE_LABELS"})
    elif "classification" in true_labels_df.columns:
        merged = merged.rename(columns={"classification": "TRUE_LABELS"})

    if "classification_x" in merged.columns:
        merged = merged.drop(columns=["classification_x"])

    if "Probability" in merged.columns:
        merged["Probability"] = pd.to_numeric(merged["Probability"], errors="coerce")
        merged["Probability"] = merged["Probability"].fillna(merged["Probability"].mean())

    if "SorfAnnotation" in merged.columns:
        mode_val = merged["SorfAnnotation"].mode()
        fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
        merged["SorfAnnotation"] = merged["SorfAnnotation"].fillna(fill_val)

    merged.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")
    print("Columns:", list(merged.columns))
    print("Rows:", len(merged))


if __name__ == "__main__":
    main()
