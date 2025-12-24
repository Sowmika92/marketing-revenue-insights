# scripts/utils.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def standardize_colnames(df):
    """
    Lowercase column names and replace spaces with underscores.
    Returns a new DataFrame (or view) with standardized columns.
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def safe_log_series(s, eps=1e-6):
    """Return log(1 + x) safe for nonpositive values by shifting if needed."""
    s = s.copy().astype(float)
    minval = s.min()
    if np.isfinite(minval) and minval <= -1:
        shift = abs(minval) + 1 + eps
    else:
        shift = 0
    return np.log1p(s + shift)


def calc_vif(df, feature_cols):
    """
    Calculate VIF for a set of features.
    Returns a dict of {feature: vif}.
    """
    df2 = df[feature_cols].dropna()
    if df2.shape[0] == 0:
        return {c: None for c in feature_cols}
    X = add_constant(df2)
    vifs = {}
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        try:
            vif = variance_inflation_factor(X.values, i + 0)  # variance_inflation_factor expects numpy array
            vifs[col] = float(vif)
        except Exception:
            vifs[col] = None
    return vifs
