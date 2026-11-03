import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
]
NUMERIC_COLUMNS = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week",
]
CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]


def make_label(df: pd.DataFrame) -> pd.Series:

    if "income" not in df.columns:
        raise KeyError(
            f"'income' column not found. Available columns: {list(df.columns)}"
        )
    return (
        df["income"]
        .astype(str)
        .str.strip()
        .str.rstrip(".")
        .eq(">50K")
        .astype(int)
    )


def build_preprocessor(extra_numeric=None) -> ColumnTransformer:

    extra_numeric = extra_numeric or []
    numeric_cols = list(NUMERIC_COLUMNS) + list(extra_numeric)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, CATEGORICAL_COLUMNS),
    ])


def clip_outliers(X_train, X_test, cols=None, percentiles=(1, 99)):
    cols = cols or NUMERIC_COLUMNS
    bounds = {}
    X_train = X_train.copy()
    X_test = X_test.copy()
    for col in cols:
        lo = np.percentile(X_train[col].astype(float), percentiles[0])
        hi = np.percentile(X_train[col].astype(float), percentiles[1])
        bounds[col] = (float(lo), float(hi))
        X_train[col] = X_train[col].astype(float).clip(lo, hi)
        X_test[col] = X_test[col].astype(float).clip(lo, hi)
    return X_train, X_test, bounds


def split_and_clean(df, test_size, seed):
    X = df[FEATURE_COLUMNS].copy()
    y = make_label(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed,
    )

    before = len(X_train)
    train_joined = pd.concat([X_train, y_train.rename("label")], axis=1)
    train_dedup = train_joined.drop_duplicates(keep="first")
    X_train = train_dedup[FEATURE_COLUMNS].reset_index(drop=True)
    y_train = train_dedup["label"].reset_index(drop=True)
    print(f"[preprocessing] duplicates dropped from train: {before - len(X_train)}")

    X_train, X_test, bounds = clip_outliers(X_train, X_test)
    print("[preprocessing] outlier bounds (train):", bounds)

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return {
        "preprocessor": preprocessor,
        "X_train": X_train_proc,
        "X_test": X_test_proc,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "bounds": bounds,
    }