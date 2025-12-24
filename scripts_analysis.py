#!/usr/bin/env python3
"""
scripts/analysis.py

End-to-end sales vs marketing analysis:
- load and clean data
- optional resampling (daily/weekly/monthly)
- descriptive stats and time-series plots
- Pearson & Spearman correlations
- OLS regression (robust SE)
- group comparisons (t-test / ANOVA / Kruskal-Wallis)
- Granger causality tests
- basic diagnostics (VIF, Breusch-Pagan)
- saves figures and a short summary CSV

Usage:
python scripts/analysis.py \
  --input data/raw/sales.csv \
  --date-col Date \
  --revenue-col Revenue \
  --mkt-col Marketing_Spend \
  --resample weekly \
  --output-dir outputs

Requirements: see requirements.txt
"""
import argparse
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import grangercausalitytests

from scripts.utils import (
    ensure_dir,
    safe_log_series,
    calc_vif,
    standardize_colnames,
)

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


def load_data(path, date_col="Date", parse_dates=True):
    df = pd.read_csv(path)
    df = standardize_colnames(df)
    if parse_dates and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df


def resample_df(df, date_col="date", freq=None, agg="sum"):
    if freq is None:
        return df
    df = df.set_index(date_col)
    if agg == "sum":
        out = df.resample(freq).sum(min_count=1)
    elif agg == "mean":
        out = df.resample(freq).mean()
    else:
        raise ValueError("agg must be 'sum' or 'mean'")
    out = out.reset_index()
    return out


def describe_and_save(df, out_dir):
    desc = df.describe(include="all").transpose()
    desc.to_csv(os.path.join(out_dir, "data_describe.csv"))
    return desc


def plot_time_series(df, date_col, revenue_col, mkt_col, out_dir):
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    sns.lineplot(data=df, x=date_col, y=revenue_col, ax=ax[0], marker="o")
    ax[0].set_title("Revenue over time")
    sns.lineplot(data=df, x=date_col, y=mkt_col, ax=ax[1], color="C1", marker="o")
    ax[1].set_title("Marketing Spend over time")
    plt.tight_layout()
    p = os.path.join(out_dir, "time_series.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def scatter_and_regplot(df, revenue_col, mkt_col, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x=mkt_col, y=revenue_col, data=df, ax=ax, scatter_kws={"s": 30, "alpha": 0.6})
    ax.set_title("Revenue vs Marketing Spend")
    plt.tight_layout()
    p = os.path.join(out_dir, "scatter_reg.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def correlation_tests(df, revenue_col, mkt_col):
    x = df[mkt_col].dropna()
    y = df[revenue_col].dropna()
    # align indices
    both = df[[mkt_col, revenue_col]].dropna()
    pearson_r, pearson_p = stats.pearsonr(both[mkt_col], both[revenue_col])
    spearman_rho, spearman_p = stats.spearmanr(both[mkt_col], both[revenue_col])
    out = {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "n": len(both),
    }
    return out


def ols_regression(df, revenue_col, mkt_col, add_const=True, robust=True):
    df2 = df[[revenue_col, mkt_col]].dropna()
    X = df2[[mkt_col]]
    if add_const:
        X = sm.add_constant(X)
    model = sm.OLS(df2[revenue_col], X).fit()
    if robust:
        res = model.get_robustcov_results(cov_type="HC3")
    else:
        res = model
    coef = res.params.to_dict()
    pvalues = res.pvalues.to_dict()
    conf_int = res.conf_int().to_dict()
    rsq = float(res.rsquared)
    summary = {
        "params": coef,
        "pvalues": pvalues,
        "conf_int": conf_int,
        "rsquared": rsq,
        "aic": float(res.aic),
        "nobs": int(res.nobs),
        "model_summary": res.summary().as_text(),
    }
    return summary, res


def heteroskedasticity_test(res):
    lm, lm_p, f_stat, f_p = het_breuschpagan(res.resid, res.model.exog)
    return {"bp_lm": float(lm), "bp_pvalue": float(lm_p), "bp_f": float(f_stat), "bp_f_pvalue": float(f_p)}


def granger_test(df, revenue_col, mkt_col, maxlag=4):
    ts = df[[revenue_col, mkt_col]].dropna()
    # the grangercausalitytests expects array: [y, x]
    # Tests whether mkt causes revenue
    results = {}
    try:
        gc_res = grangercausalitytests(ts[[revenue_col, mkt_col]], maxlag=maxlag, verbose=False)
        for lag, res in gc_res.items():
            # use ssr_ftest pvalue and/or params_ftest
            ssr_ftest_p = res[0]["ssr_ftest"][1]
            params_ftest_p = res[0]["params_ftest"][1] if "params_ftest" in res[0] else None
            results[f"lag_{lag}_ssr_ftest_p"] = float(ssr_ftest_p)
            if params_ftest_p is not None:
                results[f"lag_{lag}_params_ftest_p"] = float(params_ftest_p)
    except Exception as e:
        results["error"] = str(e)
    return results


def group_comparisons(df, revenue_col, group_col):
    out = {}
    if group_col not in df.columns:
        return {"error": f"{group_col} not in dataframe"}
    groups = df[[group_col, revenue_col]].dropna().groupby(group_col)[revenue_col]
    sizes = groups.size().to_dict()
    out["group_sizes"] = sizes
    unique_groups = df[group_col].dropna().unique()
    if len(unique_groups) == 2:
        gvals = [df[df[group_col] == g][revenue_col].dropna() for g in unique_groups]
        tstat, pval = stats.ttest_ind(gvals[0], gvals[1], equal_var=False)
        out["t_test"] = {"tstat": float(tstat), "pvalue": float(pval), "groups": list(unique_groups)}
    elif len(unique_groups) > 2:
        samples = [df[df[group_col] == g][revenue_col].dropna() for g in unique_groups]
        try:
            fstat, pval = stats.f_oneway(*samples)
            out["anova"] = {"fstat": float(fstat), "pvalue": float(pval)}
        except Exception:
            # fallback nonparametric
            kwstat, kw_p = stats.kruskal(*samples)
            out["kruskal"] = {"stat": float(kwstat), "pvalue": float(kw_p)}
    else:
        out["error"] = "less than 2 groups"
    return out


def save_regression_diagnostics(res, out_dir, revenue_col, mkt_col):
    # residuals vs fitted
    fitted = res.fittedvalues
    resid = res.resid
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.scatterplot(x=fitted, y=resid, ax=ax[0], alpha=0.6)
    ax[0].axhline(0, color="k", ls="--", lw=0.8)
    ax[0].set_title("Residuals vs Fitted")
    # QQ-plot
    sm.qqplot(resid, line="s", ax=ax[1])
    ax[1].set_title("QQ-plot of residuals")
    plt.tight_layout()
    p = os.path.join(out_dir, "regression_diagnostics.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def save_summary(summary_dict, out_path):
    pd.Series(summary_dict).to_csv(out_path)


def main(args):
    out_dir = args.output_dir or "outputs"
    ensure_dir(out_dir)
    df = load_data(args.input, date_col=args.date_col)
    # unify lowercase column names created in standardize_colnames
    date_col = args.date_col.lower()
    revenue_col = args.revenue_col.lower()
    mkt_col = args.mkt_col.lower()

    if date_col in df.columns:
        df = df.sort_values(date_col)

    # resample mapping
    freq_map = {"daily": "D", "weekly": "W", "monthly": "M"}
    freq = None
    if args.resample:
        freq = freq_map.get(args.resample.lower(), args.resample)

    if freq and date_col in df.columns:
        df = resample_df(df, date_col=date_col, freq=freq, agg=args.agg)

    # Quick data description
    describe_and_save(df, out_dir)

    # Basic plots
    if date_col in df.columns and revenue_col in df.columns and mkt_col in df.columns:
        plot_time_series(df, date_col, revenue_col, mkt_col, out_dir)
    if revenue_col in df.columns and mkt_col in df.columns:
        scatter_and_regplot(df, revenue_col, mkt_col, out_dir)

    results = {}
    if revenue_col in df.columns and mkt_col in df.columns:
        results["correlation"] = correlation_tests(df, revenue_col, mkt_col)
        summary, res = ols_regression(df, revenue_col, mkt_col, add_const=True, robust=not args.no_robust)
        results["ols"] = summary
        # diagnostics
        try:
            results["heteroskedasticity"] = heteroskedasticity_test(res)
        except Exception as e:
            results["heteroskedasticity"] = {"error": str(e)}
        try:
            diag_path = save_regression_diagnostics(res, out_dir, revenue_col, mkt_col)
            results["diagnostic_plot"] = diag_path
        except Exception:
            pass
    # VIF (if other predictors present in future)
    try:
        results["vif"] = calc_vif(df, [mkt_col])  # currently only mkt_col given; placeholder
    except Exception as e:
        results["vif"] = {"error": str(e)}

    # Granger causality (Mkt -> Revenue)
    try:
        results["granger"] = granger_test(df, revenue_col, mkt_col, maxlag=args.maxlag)
    except Exception as e:
        results["granger"] = {"error": str(e)}

    # Group comparisons (if provided)
    if args.group_col:
        gc = args.group_col.lower()
        results["group_tests"] = group_comparisons(df, revenue_col, gc)

    # Save results
    save_summary(results, os.path.join(out_dir, "analysis_summary.json"))
    # Also write a minimal human-readable CSV for key scalars
    flat = {
        "pearson_r": results.get("correlation", {}).get("pearson_r"),
        "pearson_p": results.get("correlation", {}).get("pearson_p"),
        "spearman_rho": results.get("correlation", {}).get("spearman_rho"),
        "spearman_p": results.get("correlation", {}).get("spearman_p"),
        "ols_rsquared": results.get("ols", {}).get("rsquared"),
    }
    pd.Series(flat).to_csv(os.path.join(out_dir, "key_metrics.csv"))

    print("Analysis complete. Outputs in:", out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sales vs Marketing analysis")
    p.add_argument("--input", required=True, help="Path to input CSV")
    p.add_argument("--date-col", default="Date", help="Date column name")
    p.add_argument("--revenue-col", default="Revenue", help="Revenue column name")
    p.add_argument("--mkt-col", default="Marketing_Spend", help="Marketing spend column name")
    p.add_argument("--resample", default=None, help="Resample frequency: daily|weekly|monthly or pandas freq string")
    p.add_argument("--agg", default="sum", choices=["sum", "mean"], help="Aggregation method when resampling")
    p.add_argument("--output-dir", default="outputs", help="Directory to save outputs")
    p.add_argument("--group-col", default=None, help="Optional grouping column for group comparisons")
    p.add_argument("--maxlag", type=int, default=4, help="Max lag for Granger causality")
    p.add_argument("--no-robust", action="store_true", help="Disable robust SEs for OLS")
    args = p.parse_args()
    main(args)
