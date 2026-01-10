# small-orfs-classification-ensemble

This repository implements a machine learning pipeline for the
classification of small open reading frames (sORFs) into coding and
non-coding categories using an ensemble of predictors.

The pipeline integrates predictions from multiple sORF detection tools,
merges them with curated ground-truth labels, and trains supervised
machine learning models to evaluate classification performance.

---

## Repository Structure

small-orfs-classification-ensemble/
├── build_dataset.py
└── train_models.py


- `build_dataset.py`  
  Merges outputs from multiple sORF prediction tools, cleans and
  normalizes annotations, merges them with TRUE_LABELS, and produces the
  final dataset `merged_data.csv`.

- `train_models.py`  
  Loads `merged_data.csv`, performs one-hot encoding, splits the data
  into training and test sets, trains multiple classifiers, and reports
  performance metrics.

---

## Data Processing Pipeline

1. Outputs from several sORF prediction tools are loaded.
2. Classification labels are normalized (coding / noncoding).
3. Duplicate predictions for the same sORF are removed.
4. The dataset is merged with experimentally curated TRUE_LABELS.
5. Missing values are handled and probabilities are imputed.
6. The final table is written to `merged_data.csv`.

This file is the single source of truth used for model training.

---

## Machine Learning Models

The following classifiers are trained and evaluated:

- Decision Tree
- k-Nearest Neighbors (KNN)
- Random Forest

All models are evaluated on a 50/50 stratified train–test split using:

- Accuracy  
- Confusion Matrix  
- Precision / Recall / F1-score  
- Sensitivity (Recall of positive class)  
- Specificity  

---

## Requirements

- Python 3
- pandas
- numpy
- scikit-learn

Install dependencies:

```bash
pip install pandas numpy scikit-learn
```
---
How to Run 
python build_dataset.py
python train_models.py

