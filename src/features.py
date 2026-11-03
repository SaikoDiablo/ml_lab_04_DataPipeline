import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ENGINEERED_NUMERIC = ["capital_net", "hours_per_age"]


def engineer_features(X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame):
    def add(df):
        df = df.copy()
        df["capital_net"] = (
            df["capital-gain"].astype(float) - df["capital-loss"].astype(float)
        )
    
        denom = (df["age"].astype(float) - 16).clip(lower=1.0)
        df["hours_per_age"] = df["hours-per-week"].astype(float) / denom
        return df

    return add(X_train_raw), add(X_test_raw)


def baseline_compare(X_train_proc, X_test_proc, y_train, y_test,
                     X_train_raw, X_test_raw):

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_proc, y_train)
    pred_plain = lr.predict(X_test_proc)

    plain_acc = accuracy_score(y_test, pred_plain)
    plain_f1 = f1_score(y_test, pred_plain)

    from .preprocessing import build_preprocessor
    X_train_eng, X_test_eng = engineer_features(X_train_raw, X_test_raw)

    pre = build_preprocessor(extra_numeric=ENGINEERED_NUMERIC)  # <-- key fix
    X_train_eng_proc = pre.fit_transform(X_train_eng)
    X_test_eng_proc = pre.transform(X_test_eng)

    lr_eng = LogisticRegression(max_iter=1000, random_state=42)
    lr_eng.fit(X_train_eng_proc, y_train)
    pred_eng = lr_eng.predict(X_test_eng_proc)

    eng_acc = accuracy_score(y_test, pred_eng)
    eng_f1 = f1_score(y_test, pred_eng)

    out = {
        "without_engineered": {"accuracy": float(plain_acc), "f1": float(plain_f1)},
        "with_engineered":    {"accuracy": float(eng_acc), "f1": float(eng_f1)},
    }
    print("[features] baseline (no engineered):", out["without_engineered"])
    print("[features] baseline (with engineered):", out["with_engineered"])
    return out