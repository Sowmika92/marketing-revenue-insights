# marketing-revenue-insights

Reproducible analysis for understanding the relationship between marketing spend and revenue. This repository contains datasets, notebooks, and scripts for exploratory data analysis, hypothesis testing (correlation, group comparisons), regression modeling, time-series analysis (lags & Granger causality), diagnostics, and a short actionable report.

## Features
- Data cleaning and aggregation utilities
- Exploratory plots and distribution summaries
- Correlation analysis (Pearson & Spearman) with significance tests
- OLS regression with robust standard errors and diagnostics
- Group comparisons (t-tests, ANOVA / nonparametric alternatives)
- Time-series lag analysis and Granger causality testing
- Reproducible notebooks and a standalone analysis script
- Clear guidance on interpreting results and limitations

## Getting started

Prerequisites
- Python 3.8+ recommended
- Conda or venv for environment management

Install dependencies
```bash
# using pip
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Or with conda:
```bash
conda env create -f environment.yml
conda activate marketing-revenue
```

## Data expectations
Place raw datasets in `data/raw/`. Recommended CSV column names:
- Date — ISO format (YYYY-MM-DD)
- Revenue — numeric (total revenue for the period)
- Marketing_Spend — numeric (total marketing spend for the period)
Optional:
- Channel — categorical (e.g., "Email", "Paid Search")
- Region — categorical
- Units_Sold — numeric
- Campaign_Flag — 0/1 flag for campaign periods

If your column names differ, update the notebook or `scripts/analysis.py` variable mappings.

## Typical workflow

1. Add raw data to `data/raw/`.
2. Run data cleaning & aggregation:
   - From notebook: open `notebooks/01-exploration.ipynb`
   - Or run the script:
     ```bash
     python scripts/analysis.py --input data/raw/sales.csv --output data/processed/cleaned.csv
     ```
3. Explore correlations and plots in `notebooks/02-correlation-regression.ipynb`.
4. Run time-series tests in `notebooks/03-time-series-granger.ipynb`.
5. Export a summary report to `reports/summary_report.md`.

## Notebooks
- notebooks/01-exploration.ipynb — data checks, summary statistics, and EDA
- notebooks/02-correlation-regression.ipynb — correlation tests, scatterplots, OLS regression
- notebooks/03-time-series-granger.ipynb — lagged models, Granger causality, forecasting checks

## Scripts
- scripts/analysis.py — end-to-end script to run standard analyses non-interactively
- scripts/utils.py — helper functions for cleaning, aggregation, plotting

Example script usage:
```bash
python scripts/analysis.py \
  --input data/raw/sales.csv \
  --date-col Date \
  --revenue-col Revenue \
  --mkt-col Marketing_Spend \
  --resample weekly \
  --output data/processed/cleaned.csv
```

## Output
- `data/processed/` — cleaned / aggregated datasets
- `reports/figures/` — generated plots (time series, scatterplots, residuals)
- `reports/summary_report.md` — concise findings, test results, p-values, CI, and recommendations

## Analysis summary (what to expect)
- Pearson and Spearman correlation coefficients with p-values
- Regression coefficient (slope), standard errors, R², and 95% CI
- Heteroskedasticity tests (Breusch–Pagan) and robust SEs if needed
- t-tests / ANOVA (or nonparametric alternatives) for group comparisons
- Granger causality results for lagged predictive relationships
- Diagnostics: residual plots, VIF for multicollinearity, recommendations for transformations

## Interpretation & limitations
- Correlation is not causation — confounders and seasonality can bias estimates.
- Aggregation level affects results (daily vs. weekly vs. monthly).
- Outliers and heteroskedasticity may distort Pearson/OLS results — we provide robust options.

## Contributing
Contributions welcome. Suggested workflow:
1. Fork the repo
2. Create a feature branch
3. Open a PR with tests/notebooks demonstrating the change

## License
MIT — see LICENSE file.

## Contact
If you need customization, additional metrics (e.g., ROI, LTV), or help interpreting results, open an issue or contact the maintainer.
