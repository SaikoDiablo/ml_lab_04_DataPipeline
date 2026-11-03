import json
from pathlib import Path

import pandas as pd

SCHEMA = {
    "age":              {"type": "int",      "range": (17, 90)},
    "workclass":        {"type": "category", "allowed": None},
    "fnlwgt":           {"type": "int",      "range": None},
    "education":        {"type": "category", "allowed": None},
    "education-num":    {"type": "int",      "range": None},
    "marital-status":   {"type": "category", "allowed": None},
    "occupation":       {"type": "category", "allowed": None},
    "relationship":     {"type": "category", "allowed": None},
    "race":             {"type": "category", "allowed": None},
    "sex":              {"type": "category", "allowed": None},
    "capital-gain":     {"type": "int",      "range": (0, 99999)},
    "capital-loss":     {"type": "int",      "range": (0, 4356)},
    "hours-per-week":   {"type": "int",      "range": (1, 99)},
    "native-country":   {"type": "category", "allowed": None},
    "income":           {"type": "category", "allowed": None},
}

ALLOWED_CATEGORIES = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                  "Local-gov", "State-gov", "Without-pay", "Never-worked"],
    "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
                  "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
                  "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married",
                       "Separated", "Widowed", "Married-spouse-absent",
                       "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales",
                   "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                   "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                   "Transport-moving", "Priv-house-serv", "Protective-serv",
                   "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family",
                     "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Male", "Female"],
    "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico",
                       "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India",
                       "Japan", "Greece", "South", "China", "Cuba", "Iran",
                       "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
                       "Vietnam", "Mexico", "Portugal", "Ireland", "France",
                       "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti",
                       "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland",
                       "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago",
                       "Peru", "Hong", "Holand-Netherlands"],
    "income": ["<=50K", ">50K"],
}


def _type_violations(df: pd.DataFrame, col: str, expected: str) -> int:
    if expected == "int":
        return int(df[col].isna().sum())  # NaN will be imputed later; not a type error per se
    return 0


def _range_violations(df: pd.DataFrame, col: str, lo: int, hi: int) -> int:
    s = pd.to_numeric(df[col], errors="coerce")
    return int(((s < lo) | (s > hi)).sum())


def _category_violations(df: pd.DataFrame, col: str, allowed) -> int:
    if allowed is None:
        return 0
    # NaN (from '?') is an allowed value by the null policy.
    mask = df[col].notna() & ~df[col].astype(str).isin(allowed)
    return int(mask.sum())


def validate(df: pd.DataFrame, report_path: Path) -> dict:
    report = {"n_rows": int(len(df)), "checks": {}}

    for col, spec in SCHEMA.items():
        if col not in df.columns:
            report["checks"][col] = {"present": False}
            continue

        entry = {"present": True, "type_expected": spec["type"]}
        entry["missing_count"] = int(df[col].isna().sum())

        if spec["type"] == "int":
            entry["type_violations"] = _type_violations(df, col, "int")
            if spec["range"] is not None:
                lo, hi = spec["range"]
                entry["range"] = {"min": lo, "max": hi}
                entry["range_violations"] = _range_violations(df, col, lo, hi)
        else:
            allowed = ALLOWED_CATEGORIES.get(col, spec.get("allowed"))
            entry["allowed_size"] = None if allowed is None else len(allowed)
            entry["category_violations"] = _category_violations(df, col, allowed)

        report["checks"][col] = entry

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[validation] report written to", report_path)
    for col, entry in report["checks"].items():
        bad = entry.get("range_violations", 0) + entry.get("category_violations", 0)
        if bad:
            print(f"[validation] {col}: {bad} violations")

    return report