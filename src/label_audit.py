import pandas as pd

FEATURE_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
]


def duplicate_conflict_audit(df: pd.DataFrame) -> dict:
    grouped = df.groupby(FEATURE_COLUMNS, dropna=False)["income"].nunique()
    conflicting_groups = grouped[grouped > 1]
    n_conflicts = int(conflicting_groups.sum())
    print(f"[label_audit] duplicate feature rows with conflicting income: "
          f"{n_conflicts} groups")
    return {"conflicting_groups": int(len(conflicting_groups)),
            "conflicting_rows": n_conflicts}


def manual_inspection(df: pd.DataFrame, n: int = 50, seed: int = 42) -> pd.DataFrame:
    sample = df.sample(n=n, random_state=seed)

    impossible = []
    for idx, row in sample.iterrows():
        problems = []
        if pd.notna(row["age"]) and not (17 <= row["age"] <= 90):
            problems.append("age out of [17,90]")
        if pd.notna(row["hours-per-week"]) and not (1 <= row["hours-per-week"] <= 99):
            problems.append("hours out of [1,99]")
        if pd.notna(row["capital-gain"]) and row["capital-gain"] < 0:
            problems.append("negative capital-gain")
        if pd.notna(row["capital-loss"]) and row["capital-loss"] < 0:
            problems.append("negative capital-loss")
        if problems:
            impossible.append({"index": int(idx), "issues": problems})

    print(f"[label_audit] manual inspection: {len(impossible)}/{n} rows with issues")
    return pd.DataFrame(impossible)