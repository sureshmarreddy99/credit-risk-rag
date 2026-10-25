"""Train six leakage-aware baseline classifiers on LendingClub accepted loans.

Usage: python src/model_a/train_models.py --csv data/raw/accepted_2007_to_2018Q4.csv
The target is terminal Charged Off vs Fully Paid, NOT 12-month default.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib

# Strict allowlist: only fields plausibly known at application/origination.
FEATURES = [
    "loan_amnt", "term", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "dti", "delinq_2yrs", "earliest_cr_line",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
    "pub_rec_bankruptcies", "application_type", "addr_state"
]
CAT = {"term", "emp_length", "home_ownership", "verification_status",
       "purpose", "application_type", "addr_state"}
def load_data(path, max_rows=None):
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    required = {"loan_status", "issue_d"}
    if not required.issubset(columns):
        raise ValueError(f"Missing required columns: {required - set(columns)}")
    use = [c for c in ["loan_status", "issue_d"] + FEATURES if c in columns]
    df = pd.read_csv(path, usecols=use, low_memory=False, nrows=max_rows)
    df = df[df.loan_status.isin(["Fully Paid", "Charged Off"])].copy()
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df.dropna(subset=["issue_d"])
    df["target"] = (df.loan_status == "Charged Off").astype(int)
    if "earliest_cr_line" in df:
        first = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
        df["credit_history_months"] = ((df.issue_d - first).dt.days / 30.44).clip(lower=0)
        df = df.drop(columns="earliest_cr_line")
    for col in ["revol_util"]:
        if col in df:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", "", regex=False), errors="coerce")
    for col in df.columns:
        if col not in CAT and col not in {"loan_status", "issue_d", "target"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("issue_d").reset_index(drop=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="results/model_a")
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df = load_data(args.csv, args.max_rows)
    if len(df) < 100 or df.target.nunique() != 2:
        raise ValueError("Not enough labeled observations for training.")
    # Out-of-time holdout: final 20% by origination date, with no shared month.
    months = sorted(df.issue_d.unique())
    if len(months) < 5:
        raise ValueError("At least five distinct origination months are needed.")
    cutoff = months[int(len(months) * .8)]
    train = df[df.issue_d < cutoff].copy()
    test = df[df.issue_d >= cutoff].copy()
    if train.target.nunique() < 2 or test.target.nunique() < 2:
        raise ValueError("Both classes must be present in train and test.")
    columns = [c for c in df.columns if c not in {"loan_status", "issue_d", "target"}]
    cats = [c for c in columns if c in CAT]
    nums = [c for c in columns if c not in CAT]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), nums),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats)
    ])
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=100, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(n_estimators=350, max_depth=4, learning_rate=.05, subsample=.8, colsample_bytree=.8, tree_method="hist", n_jobs=-1, random_state=42)
    except ImportError:
        print("Skipping XGBoost: install xgboost")
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(n_estimators=350, learning_rate=.05, num_leaves=31, verbose=-1, random_state=42)
    except ImportError:
        print("Skipping LightGBM: install lightgbm")
    try:
        from catboost import CatBoostClassifier
        models["catboost"] = CatBoostClassifier(iterations=350, depth=6, learning_rate=.05, verbose=False, random_seed=42)
    except ImportError:
        print("Skipping CatBoost: install catboost")
    results = []
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocess", prep), ("model", estimator)])
        pipeline.fit(train[columns], train.target)
        prob = pipeline.predict_proba(test[columns])[:, 1]
        pred = (prob >= .5).astype(int)
        result = dict(model=name, roc_auc=roc_auc_score(test.target, prob),
            pr_auc=average_precision_score(test.target, prob),
            brier=brier_score_loss(test.target, prob),
            precision=precision_score(test.target, pred, zero_division=0),
            recall=recall_score(test.target, pred, zero_division=0),
            f1=f1_score(test.target, pred, zero_division=0))
        results.append(result)
        print(result, flush=True)
        joblib.dump(pipeline, out / f"{name}.joblib")
    pd.DataFrame(results).sort_values("roc_auc", ascending=False).to_csv(out / "metrics.csv", index=False)
    (out / "run_metadata.json").write_text(json.dumps({
        "train_rows": len(train), "test_rows": len(test), "cutoff_month": str(cutoff),
        "target_definition": "Charged Off versus Fully Paid, terminal outcomes",
        "feature_columns": columns, "warning": "Outcome-maturity bias may remain; not a 12-month PD model."
    }, indent=2))
if __name__ == "__main__":
    main()
