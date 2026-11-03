import pandas as pd
from sklearn.datasets import fetch_openml

def load_raw_adult(version: int = 2) -> pd.DataFrame:
    bunch = fetch_openml(name="adult", version=version, as_frame=True)
    df = bunch.frame.copy()

    n_rows, n_cols = df.shape
    print(f"[ingestion] raw rows = {n_rows}")
    print(f"[ingestion] raw cols = {n_cols}")
    assert n_rows == 48842, f"Unexpected row count: {n_rows}"
    assert n_cols == 15, f"Unexpected column count: {n_cols}"

    df = df.replace("?", pd.NA)

 
    if "class" in df.columns and "income" not in df.columns:
        df = df.rename(columns={"class": "income"})

    return df