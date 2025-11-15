# ==========================================
# 📊 Dynamic Portfolio Optimization Dashboard
# ==========================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# ------------------------------------------
# Rolling Portfolio Rebalance Function
# ------------------------------------------
def rolling_rebalance(
    returns_data,
    rebalance_freq=100,
    lookback_min=100,
    objective='sharpe',
    l2_reg=1e-4,
    allow_short=False,
    start_cash=1.0,
    trading_days=252,
    target_vol=0.20,
    vol_window=100,
    verbose=False
):
    returns_df = pd.DataFrame({t: pd.Series(v) for t, v in returns_data.items()})
    returns_df = returns_df.dropna(how='all', axis=1).fillna(0.0)
    tickers = list(returns_df.columns)
    n_steps = len(returns_df)
    if n_steps == 0:
        raise ValueError("returns_data empty or invalid.")

    port_returns = np.zeros(n_steps)
    applied_weights = np.zeros((n_steps, len(tickers)))
    weight_history, weights_index, vol_history = [], [], []
    last_weights = np.repeat(1.0 / len(tickers), len(tickers))
    step = 0

    while step < n_steps:
        hist_end = step
        if hist_end < lookback_min:
            w = last_weights
            scale = 1.0
            realized_vol = np.nan
        else:
            hist = returns_df.iloc[:hist_end].copy()
            mean_ret = hist.mean().values
            cov = hist.cov().values
            inv_cov = np.linalg.pinv(cov)

            if objective == "sharpe":
                w = inv_cov @ mean_ret
            elif objective == "drawdown":
                w = 1 / np.sqrt(np.diag(cov))
            else:  # hybrid
                w = 0.7 * (inv_cov @ mean_ret) + 0.3 * (1 / np.sqrt(np.diag(cov)))

            w = np.maximum(w, 0)
            w = w / w.sum()

            # Volatility Targeting
            recent_returns = hist.values.dot(w)
            realized_vol = np.std(recent_returns[-vol_window:], ddof=0) * np.sqrt(trading_days)
            scale = np.clip(target_vol / max(1e-8, realized_vol), 0.5, 1.5)
            w = w * scale
            w = w / w.sum()

        end_apply = min(step + rebalance_freq, n_steps)
        seg_idx = np.arange(step, end_apply)
        seg_returns = returns_df.iloc[seg_idx].values.dot(w)
        port_returns[seg_idx] = seg_returns
        applied_weights[seg_idx, :] = w
        weight_history.append(pd.Series(w, index=tickers))
        weights_index.append(step)

        if not np.isnan(realized_vol):
            vol_history.append({'step': step, 'realized_vol': realized_vol, 'scale': scale, 'target_vol': target_vol})

        last_weights = w
        step = end_apply

    # --- Metrics Calculation (Fixed & Safe) ---
    port_returns = np.clip(port_returns, -0.99, 1.0)
    equity = start_cash * np.cumprod(1 + port_returns)
    equity_series = pd.Series(equity)

    mean_step = np.mean(port_returns)
    std_step = np.std(port_returns, ddof=0)
    sharpe = (mean_step / (std_step + 1e-9)) * np.sqrt(trading_days)
    years = len(port_returns) / trading_days
    cum_return = equity[-1] / equity[0] - 1
    cagr = (equity[-1]) ** (1 / max(1e-9, years)) - 1

    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd = drawdown.min()

    weights_df = pd.DataFrame(weight_history, index=weights_index)
    vol_df = pd.DataFrame(vol_history)

    return {
        'equity_series': equity_series,
        'port_returns': port_returns,
        'weights_df': weights_df,
        'metrics': {
            'Sharpe': sharpe,
            'CAGR': cagr,
            'CumReturn': cum_return,
            'MaxDD': max_dd
        },
        'vol_history': vol_df,
        'drawdown': drawdown,
        'returns_df': returns_df
    }

# ------------------------------------------
# Streamlit UI
# ------------------------------------------
st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")
st.title("💹 Dynamic Portfolio Optimization Dashboard")

st.markdown("""
This dashboard dynamically simulates **rolling rebalancing portfolio strategies** using historical data.  
Adjust parameters below to explore risk-adjusted optimization (Sharpe, Drawdown, or Hybrid).
""")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
tickers = st.sidebar.text_input("Stock Tickers (comma separated)", "AAPL,MSFT,NVDA,GOOG").split(",")
tickers = [t.strip().upper() for t in tickers if t.strip()]
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))
rebalance_freq = st.sidebar.slider("Rebalance Frequency", 50, 300, 100, step=25)
lookback_min = st.sidebar.slider("Lookback Window", 50, 300, 120, step=10)
objective = st.sidebar.selectbox("Optimization Objective", ["sharpe", "drawdown", "hybrid"])
target_vol = st.sidebar.slider("Target Volatility (%)", 5, 40, 20, step=1)
trading_days = st.sidebar.number_input("Trading Days per Year", 100, 365, 252)
run_btn = st.sidebar.button("🚀 Run Simulation")

# ------------------------------------------
# Load Historical Data
# ------------------------------------------
@st.cache_data
def load_data(tickers, start_date, end_date):
    raw = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw["Adj Close"] if "Adj Close" in raw.columns.levels[0] else raw["Close"]
    else:
        data = raw.get("Adj Close", raw.get("Close"))
    returns = data.pct_change().dropna()
    return data, returns

data, returns_df = load_data(tickers, start_date, end_date)

if data.empty:
    st.error("⚠️ No valid data found for the selected tickers.")
    st.stop()

st.line_chart(data)
st.caption("📈 Historical adjusted closing prices of selected stocks.")

# ------------------------------------------
# Run Simulation
# ------------------------------------------
if run_btn:
    st.subheader("📊 Portfolio Optimization Results")
    results = rolling_rebalance(
        returns_data={t: returns_df[t].values for t in returns_df.columns},
        rebalance_freq=rebalance_freq,
        lookback_min=lookback_min,
        objective=objective,
        target_vol=target_vol / 100,
        trading_days=trading_days,
        verbose=False
    )

    metrics = results["metrics"]
    eq = results["equity_series"]
    drawdown = results["drawdown"]
    weights_df = results["weights_df"]
    vol_df = results["vol_history"]
    corr = results["returns_df"].corr()

    # 1️⃣ Metrics Summary
    st.markdown("### 📈 Performance Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpe Ratio", f"{metrics['Sharpe']:.2f}")
    col2.metric("CAGR", f"{metrics['CAGR']*100:.2f}%")
    col3.metric("Max Drawdown", f"{metrics['MaxDD']*100:.2f}%")

    # 2️⃣ Equity Curve
    st.markdown("### 💹 Portfolio Growth Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eq.index, eq / eq.iloc[0], color="gold", linewidth=2)
    ax.set_title(f"{objective.capitalize()} Optimized Portfolio")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.4)
    st.pyplot(fig)

    # 3️⃣ Drawdown Visualization
    st.markdown("### 📉 Drawdown (Peak-to-Trough Loss Over Time)")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(range(len(drawdown)), drawdown * 100, color="red", alpha=0.3)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # 4️⃣ Portfolio Weights
    st.markdown("### 🧩 Portfolio Weight Allocation Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    weights_df.plot(ax=ax, linewidth=1.2)
    ax.set_xlabel("Rebalance Step")
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    # 5️⃣ Volatility Targeting
    if not vol_df.empty:
        st.markdown("### ⚖️ Volatility Targeting Dynamics")
        fig, ax1 = plt.subplots(figsize=(10, 3))
        ax2 = ax1.twinx()
        ax1.plot(vol_df['step'], vol_df['realized_vol'] * 100, 'r-', label='Realized Vol (%)')
        ax1.plot(vol_df['step'], vol_df['target_vol'] * 100, 'b--', label='Target Vol (%)')
        ax2.plot(vol_df['step'], vol_df['scale'], 'g-', linewidth=2, label='Scale Factor')
        ax1.set_ylabel("Volatility (%)")
        ax2.set_ylabel("Scale Factor")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig)

    # 6️⃣ Correlation Heatmap
    st.markdown("### 🔥 Asset Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # 7️⃣ Insights Section
    st.markdown("## 🧠 Insights & Explanation")
    st.info(f"""
    **Strategy Used:** `{objective.capitalize()}`
    - **Sharpe Ratio**: {metrics['Sharpe']:.2f} → Return per unit of risk.  
    - **CAGR**: {metrics['CAGR']*100:.2f}% → Annualized growth rate of the portfolio.  
    - **Max Drawdown**: {metrics['MaxDD']*100:.2f}% → Largest loss from a peak.  
    - **Target Volatility:** {target_vol:.2f}  
    - **Rebalance Frequency:** {rebalance_freq} steps  
    - **Lookback Window:** {lookback_min} steps  

    💡 *Interpretation:*  
    - A **higher Sharpe** indicates better risk-adjusted returns.  
    - A **lower drawdown** means smoother, more stable performance.  
    - **Hybrid** optimization combines both return-seeking (Sharpe) and risk control (Drawdown).
    """)
else:
    st.info("👆 Configure parameters and click **Run Simulation** to start.")
