"""
Hybrid Quantamental Multi-Agent Investment System
FIN 580 Final Project — Arielle Ward & Qiaoqi Li
Run: python main.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE    = "final_dataset.csv"
RESULTS_DIR  = "results"
TOP_N        = 10
INITIAL_CAP  = 1_000_000
TRANS_COST   = 0.003   # 30 bps round-trip
RISK_FREE    = 0.02
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load & Prep ───────────────────────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    # forward-fill fundamentals within each ticker
    df[["PE","ROE","PB"]] = df.groupby("ticker")[["PE","ROE","PB"]].ffill()
    df[["PE","ROE","PB"]] = df.groupby("ticker")[["PE","ROE","PB"]].bfill()
    df["momentum"] = df.groupby("ticker")["momentum"].ffill()
    return df

# ── Helpers ───────────────────────────────────────────────────────────────────
def cross_rank(series):
    """Cross-sectional percentile rank scaled to [0,1]."""
    return series.rank(pct=True)

def compound_monthly(df):
    """Compound daily returns to monthly returns per ticker."""
    df = df.copy()
    df["ym"] = df["date"].dt.to_period("M")
    monthly = (
        df.groupby(["ym", "ticker"])["return"]
        .apply(lambda x: (1 + x.fillna(0)).prod() - 1)
        .reset_index()
    )
    monthly.columns = ["ym", "ticker", "monthly_return"]
    return monthly

def get_rebal_dates(df):
    """Last trading day of each month."""
    df["ym"] = df["date"].dt.to_period("M")
    return df.groupby("ym")["date"].max().reset_index()

# ── Agents ────────────────────────────────────────────────────────────────────
def value_agent(snap):
    """Low PE, high ROE, low PB → high value_score."""
    s = pd.DataFrame(index=snap.index)
    s["v1"] = cross_rank(-snap["PE"])
    s["v2"] = cross_rank(snap["ROE"])
    s["v3"] = cross_rank(-snap["PB"])
    return s.mean(axis=1).rename("value_score")

def momentum_agent(snap):
    """High momentum → high momentum_score."""
    return cross_rank(snap["momentum"]).rename("momentum_score")

def narrative_agent(month_sentiment):
    """Cross-sectional rank of mean monthly sentiment."""
    return cross_rank(month_sentiment["sentiment"]).rename("sentiment_score")

def verification_agent(scores):
    """Penalise or boost sentiment based on consistency with value."""
    adj = scores["sentiment_score"].copy()
    # Overconfident positive sentiment despite weak fundamentals
    mask_penalty = (scores["sentiment_score"] > 0.6) & (scores["value_score"] < 0.3)
    adj[mask_penalty] *= 0.5
    # Overly pessimistic sentiment despite strong fundamentals
    mask_boost   = (scores["sentiment_score"] < 0.4) & (scores["value_score"] > 0.7)
    adj[mask_boost] = (adj[mask_boost] * 1.3).clip(upper=1.0)
    return adj.rename("verified_sentiment")

def portfolio_manager(scores, w_val, w_mom, w_sent):
    """Weighted aggregation → top-N selection."""
    scores = scores.copy()
    scores["final_score"] = (
        w_val  * scores["value_score"] +
        w_mom  * scores["momentum_score"] +
        w_sent * scores["verified_sentiment"]
    )
    selected = scores.nlargest(TOP_N, "final_score").index
    return selected

# ── Signal Builder ────────────────────────────────────────────────────────────
def build_signals(df):
    """
    For each month-end snapshot, compute all agent scores.
    Returns a dict: {period -> list of selected tickers}
    """
    rebal = get_rebal_dates(df)
    monthly_ret = compound_monthly(df)
    
    # Monthly mean sentiment per ticker
    df["ym"] = df["date"].dt.to_period("M")
    monthly_sent = (
        df.groupby(["ym", "ticker"])["sentiment"]
        .mean()
        .reset_index()
    )

    records = []
    for _, row in rebal.iterrows():
        ym   = row["ym"]
        date = row["date"]

        snap = df[df["date"] == date].set_index("ticker")
        if len(snap) < TOP_N:
            continue

        sent_snap = monthly_sent[monthly_sent["ym"] == ym].set_index("ticker")
        if sent_snap.empty:
            continue

        # align indices
        tickers = snap.index.intersection(sent_snap.index)
        snap      = snap.loc[tickers]
        sent_snap = sent_snap.loc[tickers]

        vs  = value_agent(snap)
        ms  = momentum_agent(snap)
        ns  = narrative_agent(sent_snap)

        scores = pd.concat([vs, ms, ns], axis=1).dropna()
        vs_s = scores["value_score"]
        ms_s = scores["momentum_score"]

        verified = verification_agent(scores)
        scores["verified_sentiment"] = verified

        records.append({
            "ym": ym,
            "scores": scores
        })

    return records, monthly_ret

# ── Backtest Engine ───────────────────────────────────────────────────────────
def backtest(records, monthly_ret, w_val, w_mom, w_sent):
    """
    Given agent weight config, run full backtest.
    Signals at period T → invest in period T+1.
    """
    period_list = [r["ym"] for r in records]
    portfolio_returns = []

    for i, rec in enumerate(records):
        ym     = rec["ym"]
        scores = rec["scores"].copy()
        scores["verified_sentiment"] = verification_agent(scores)

        selected = portfolio_manager(scores, w_val, w_mom, w_sent)

        # invest in NEXT period
        next_ym = ym + 1
        next_rets = monthly_ret[
            (monthly_ret["ym"] == next_ym) &
            (monthly_ret["ticker"].isin(selected))
        ]["monthly_return"]

        if next_rets.empty:
            continue

        port_ret = next_rets.mean()  # equal weight
        port_ret -= TRANS_COST       # transaction cost
        portfolio_returns.append({"ym": next_ym, "return": port_ret})

    ret_series = pd.DataFrame(portfolio_returns).set_index("ym")["return"]
    return ret_series

# ── Baselines ─────────────────────────────────────────────────────────────────
def baseline_equal_weight(monthly_ret):
    """All 30 tickers, equal weight."""
    ew = monthly_ret.groupby("ym")["monthly_return"].mean() - TRANS_COST
    return ew.rename("return")

def baseline_momentum_only(records, monthly_ret):
    return backtest(records, monthly_ret, w_val=0.0, w_mom=1.0, w_sent=0.0)

def baseline_value_only(records, monthly_ret):
    return backtest(records, monthly_ret, w_val=1.0, w_mom=0.0, w_sent=0.0)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(ret_series, name):
    r   = ret_series.dropna()
    ann_ret  = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol  = r.std() * np.sqrt(12)
    sharpe   = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else np.nan
    cum      = (1 + r).cumprod()
    drawdown = (cum / cum.cummax() - 1)
    max_dd   = drawdown.min()
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    total    = cum.iloc[-1] - 1
    return {
        "Strategy":        name,
        "Ann. Return":     f"{ann_ret:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio":    f"{sharpe:.3f}",
        "Max Drawdown":    f"{max_dd:.2%}",
        "Calmar Ratio":    f"{calmar:.3f}",
        "Total Return":    f"{total:.2%}",
        # raw for charts
        "_sharpe": sharpe,
        "_ann_ret": ann_ret,
        "_max_dd": max_dd,
    }

def to_nav(ret_series):
    return INITIAL_CAP * (1 + ret_series.fillna(0)).cumprod()

# ── Charts ────────────────────────────────────────────────────────────────────
def plot_cumulative(nav_dict):
    fig, ax = plt.subplots(figsize=(12, 6))
    styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    for i, (name, nav) in enumerate(nav_dict.items()):
        idx = [str(p) for p in nav.index]
        ax.plot(idx, nav.values, label=name, linewidth=2, linestyle=styles[i % len(styles)])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    step = max(1, len(idx) // 8)
    ax.set_xticks(range(0, len(idx), step))
    ax.set_xticklabels([idx[i] for i in range(0, len(idx), step)], rotation=45, ha="right")
    ax.set_title("Cumulative Portfolio Value — All Strategies", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/cumulative_returns.png", dpi=150)
    plt.close()
    print("  Saved cumulative_returns.png")

def plot_drawdown(ret_dict):
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, ret in ret_dict.items():
        cum = (1 + ret.fillna(0)).cumprod()
        dd  = cum / cum.cummax() - 1
        idx = [str(p) for p in dd.index]
        ax.plot(idx, dd.values * 100, label=name, linewidth=1.5)
    step = max(1, len(idx) // 8)
    ax.set_xticks(range(0, len(idx), step))
    ax.set_xticklabels([idx[i] for i in range(0, len(idx), step)], rotation=45, ha="right")
    ax.set_title("Drawdown Over Time — All Strategies", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/drawdown_chart.png", dpi=150)
    plt.close()
    print("  Saved drawdown_chart.png")

def plot_ablation_bar(metrics_list):
    ablation_names = ["Full Model", "No Narrative", "No Value", "No Momentum"]
    sharpes = [m["_sharpe"] for m in metrics_list if m["Strategy"] in ablation_names]
    names   = [m["Strategy"] for m in metrics_list if m["Strategy"] in ablation_names]
    colors  = ["#2ecc71" if n == "Full Model" else "#e74c3c" for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, sharpes, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Ablation Study — Sharpe Ratio by Model Configuration",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio"); ax.set_ylim(0, max(sharpes) * 1.3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/ablation_bar_chart.png", dpi=150)
    plt.close()
    print("  Saved ablation_bar_chart.png")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  Hybrid Quantamental Multi-Agent Investment System")
    print("  FIN 580 — Arielle Ward & Qiaoqi Li")
    print("="*60 + "\n")

    print("Loading data...")
    df = load_data(DATA_FILE)
    print(f"  {len(df):,} rows | {df['ticker'].nunique()} tickers | "
          f"{df['date'].min().date()} → {df['date'].max().date()}\n")

    print("Building signals...")
    records, monthly_ret = build_signals(df)
    print(f"  {len(records)} rebalancing periods\n")

    # ── Run strategies ────────────────────────────────────────────────────────
    print("Running backtests...")
    strats = {}
    strats["Full Model"]     = backtest(records, monthly_ret, 0.30, 0.30, 0.40)
    strats["No Narrative"]   = backtest(records, monthly_ret, 0.50, 0.50, 0.00)
    strats["No Value"]       = backtest(records, monthly_ret, 0.00, 0.50, 0.50)
    strats["No Momentum"]    = backtest(records, monthly_ret, 0.50, 0.00, 0.50)
    strats["Equal Weight"]   = baseline_equal_weight(monthly_ret)
    strats["Momentum Only"]  = baseline_momentum_only(records, monthly_ret)
    strats["Value Only"]     = baseline_value_only(records, monthly_ret)

    # align all series to common index
    all_idx = sorted(set().union(*[s.index for s in strats.values()]))
    for k in strats:
        strats[k] = strats[k].reindex(all_idx)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics_list = [compute_metrics(v, k) for k, v in strats.items()]

    # Print table
    display_cols = ["Strategy","Ann. Return","Ann. Volatility",
                    "Sharpe Ratio","Max Drawdown","Calmar Ratio","Total Return"]
    header = f"{'Strategy':<20}" + "".join(f"{c:>16}" for c in display_cols[1:])
    print("\n" + "="*len(header))
    print(header)
    print("="*len(header))
    for m in metrics_list:
        row = f"{m['Strategy']:<20}" + "".join(f"{m[c]:>16}" for c in display_cols[1:])
        print(row)
    print("="*len(header) + "\n")

    # ── Charts ────────────────────────────────────────────────────────────────
    print("Generating charts...")
    nav_dict = {k: to_nav(v) for k, v in strats.items()}
    plot_cumulative(nav_dict)
    plot_drawdown(strats)
    plot_ablation_bar(metrics_list)

    print(f"\nAll results saved to /{RESULTS_DIR}/")
    print("Done.\n")

if __name__ == "__main__":
    main()
