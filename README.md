# Finanacial-Stock-Prediction

A compact research/demo repository showcasing stock return modeling and a dynamic portfolio optimization dashboard. The project includes a Jupyter notebook for analysis and a Streamlit dashboard that simulates rolling rebalancing strategies with volatility targeting and simple optimization objectives.

> NOTE: The repository is mostly Jupyter Notebook content. The interactive dashboard script is implemented in `portfolio_dashboard.py`.

---

## Contents

- `stock_pred.ipynb` — Main Jupyter notebook with data exploration, modeling experiments, and visualizations.
- `portfolio_dashboard.py` — Streamlit app to interactively simulate rolling rebalancing strategies across user-selected tickers and parameters.
- `apple_data/` — Optional data folder (place CSVs here for offline use).
- `logs/` — Runtime logs.
- `README.md` — This file.

---

## Key features

- Rolling rebalancing simulation with configurable:
  - Rebalance frequency and lookback window
  - Optimization objective: `sharpe`, `drawdown`, or `hybrid`
  - Volatility targeting (scale factor to target portfolio volatility)
  - Trading days per year and custom date range
- Interactive visualizations:
  - Equity curve, drawdown, and weight allocation over time
  - Volatility targeting dynamics and scale factor
  - Asset correlation heatmap
  - Performance metrics (Sharpe, CAGR, Max Drawdown)
- Uses Yahoo Finance (`yfinance`) for historical adjusted close prices.

---

## Quick start

1. Clone the repository
   ```bash
   git clone https://github.com/Sharanyea/Finanacial-Stock-Prediction.git
   cd Finanacial-Stock-Prediction
   ```

2. Create and activate a virtual environment (recommended)
   ```bash
   python -m venv venv
   # macOS/Linux
   source venv/bin/activate
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn yfinance jupyterlab
   ```

4. Run the Streamlit dashboard
   ```bash
   streamlit run portfolio_dashboard.py
   ```

5. Open the notebook
   ```bash
   jupyter lab
   # then open stock_pred.ipynb
   ```

---

## Strategy summary

The dashboard implements a rolling rebalancing routine that:

- Computes historical returns from adjusted close prices.
- On each rebalance step, estimates weights using either an inverse-covariance mean-return solution (Sharpe-style), an inverse-volatility heuristic (drawdown-focused), or a hybrid blend.
- Applies non-negative weights and normalizes them to sum to 1.
- Scales weights to meet a user-specified annualized volatility target (clipped to avoid extreme leverage).
- Advances through returns applying the most recent weights for the next rebalance interval.

Outputs include `equity_series`, `port_returns`, `weights_df`, `metrics` (Sharpe, CAGR, CumReturn, MaxDD), and `vol_history`.

---

## UI parameters

- Stock Tickers — comma-separated list (e.g., `AAPL,MSFT,NVDA`)
- Start / End Date — historical price window
- Rebalance Frequency — number of observations between rebalances
- Lookback Window — minimum historical length before optimization
- Optimization Objective — `sharpe`, `drawdown`, or `hybrid`
- Target Volatility (%) — annualized volatility target
- Trading Days per Year — used to annualize volatility and Sharpe

---

## Suggestions / Next steps

- Add a `requirements.txt` or `environment.yml` for reproducible installs.
- Add unit tests for the rebalancing logic and metrics.
- Include a small sample dataset for offline demos in `apple_data/`.
- Add a LICENSE and CONTRIBUTING if you plan to accept external contributions.

---

## License

No license file is included. Add a license (MIT, Apache-2.0, etc.) if you want to make this project open-source.

---

## Contact

Created by Sharanyea. Built with Python, Streamlit, yfinance, NumPy, pandas, Matplotlib, and Seaborn.
