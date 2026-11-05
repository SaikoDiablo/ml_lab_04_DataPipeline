# Lab 4 Report — Data Pipeline and Quality

## Goal
Build a reproducible, leakage-free pipeline for the UCI Adult dataset (predict income >50K vs ≤50K), run schema/range/missing/duplicate/label-quality checks, engineer a couple of derived features, and produce a data card plus processed-data hash so the whole thing runs with one command.

## Method
Loaded UCI Adult with `fetch_openml(name='adult', version=2, as_frame=True)`. Ingestion read the raw data untouched and logged 48,842 rows × 15 columns.

Validation checked types, ranges (age 17–90, hours 1–99, capital-gain 0–99999, capital-loss 0–4356), a `'?'` = missing null policy, and allowed categories. No rows silently dropped.

Split 70/30 stratified **before** cleaning. Imputer (mode for categorical, median for numeric), outlier clipping to 1st/99th percentile, and duplicate dropping were all fit on the train split only. Test rows never touched. Preprocessor saved to `results/preprocessor.joblib`.

Two derived features:
- `capital_net = capital-gain − capital-loss`
- `hours_per_age = hours-per-week / (age − 16)`

Ran a LogisticRegression baseline with and without them to see if they help.

Label-quality check: duplicate-conflict audit (group feature columns, count distinct income labels) plus a manual inspection of 50 random rows for impossible values.

## Results

**Ingestion and cleaning**

| Step | Result |
|---|---|
| Raw rows | 48,842 |
| Raw columns | 15 |
| Duplicates dropped from train | 24 |
| Outlier bounds (train) | age 17–74, fnlwgt 27,035–509,629, education-num 3–16, capital-gain 0–15,024, capital-loss 0–1,980, hours-per-week 8–80 |

**Baseline with vs without engineered features**

| Setup | Accuracy | F1 |
|---|---:|---:|
| No engineered features | 0.85395 | 0.66010 |
| With engineered features | **0.85778** | **0.66963** |

**Label-quality checks**

| Check | Result |
|---|---:|
| Duplicate feature rows with conflicting income | **10 groups** |
| Impossible rows in 50-row manual inspection | **0 / 50** |
