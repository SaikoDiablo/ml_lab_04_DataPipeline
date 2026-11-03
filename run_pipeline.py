import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.ingestion import load_raw_adult
from src.validation import validate
from src.preprocessing import split_and_clean
from src.features import baseline_compare
from src.label_audit import duplicate_conflict_audit, manual_inspection


def sha256_of_parquet(df: pd.DataFrame) -> str:
    """Deterministic SHA256 of a DataFrame."""
    digest = hashlib.sha256()
    digest.update(df.to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()

def write_data_card(path: Path, shape: tuple, hash_hex: str):
    card = """# Data Card - UCI Adult

## Source
UCI Machine Learning Repository, "Adult" dataset, version 2,
accessed via sklearn.datasets.fetch_openml(name='adult', version=2).

## License
CC BY 4.0 (per UCI repository metadata).

## Size
{n_rows} rows x {n_cols} columns.
SHA256 of processed sample: `{hash_hex}`

## Known biases
- Class imbalance: `<=50K` is dominant.
- Sex, race, and native-country are imbalanced and correlated with income.
- `workclass` and `occupation` include significant `'?'` missingness.
- The label is a proxy for real-world income and reflects historical
  inequality; a model trained on it can encode those biases.

## Privacy concerns
- Contains self-reported demographic attributes (sex, race, native-country).
- No direct identifiers (names, SSNs) in the released version.
- Still sensitive enough to warrant restricted redistribution and
  anonymization considerations in downstream use.

## Intended use
Educational ML lab: binary classification benchmark.
"""

    text = card.format(n_rows=shape[0], n_cols=shape[1], hash_hex=hash_hex)
    path.write_text(text, encoding="utf-8")


def main():
    cfg_path = Path("config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) Ingestion
    raw_df = load_raw_adult(version=cfg["dataset"]["version"])

    # 2) Validation
    validate(raw_df, Path(cfg["paths"]["validation_report"]))

    # 3) Split + clean (leakage-free)
    out = split_and_clean(raw_df, test_size=cfg["test_size"], seed=cfg["seed"])

    # 4) Persist preprocessor
    joblib.dump(out["preprocessor"], cfg["paths"]["preprocessor"])
    print("[pipeline] preprocessor saved to", cfg["paths"]["preprocessor"])

    # 5) Baseline: with vs without engineered features
    feat_report = baseline_compare(
        out["X_train"], out["X_test"], out["y_train"], out["y_test"],
        out["X_train_raw"], out["X_test_raw"],
    )
    with open(results_dir / "baseline_report.json", "w") as f:
        json.dump(feat_report, f, indent=2)

    # 6) Label-quality checks
    audit = duplicate_conflict_audit(raw_df)
    impossible = manual_inspection(raw_df, n=cfg["label_audit"]["n_rows"], seed=cfg["seed"])
    with open(results_dir / "label_audit.json", "w") as f:
        json.dump({"duplicate_conflict": audit,
                   "impossible_rows": impossible.to_dict(orient="records")}, f, indent=2)

    # 7) Hash of processed dataset (train+test features) and data card
    processed_sample = pd.concat(
        [out["X_train_raw"], out["X_test_raw"]], axis=0, ignore_index=True
    )
    processed_hash = sha256_of_parquet(processed_sample)
    Path(cfg["paths"]["processed_hash"]).write_text(processed_hash, encoding="utf-8")
    write_data_card(Path(cfg["paths"]["data_card"]),
                    processed_sample.shape, processed_hash)

    # 8) Save config alongside results for reproducibility
    (results_dir / "run_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8"
    )

    print("\nLab 04 completed. Artifacts in:", results_dir)


if __name__ == "__main__":
    main()