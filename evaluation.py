"""
evaluation.py
Evaluation, metrics, ablation analysis, and chart generation
for the Hybrid Quantamental Multi-Agent Investment System.

FIN 580 | Arielle Ward & Qiaoqi Li | May 2026

Usage:
    python evaluation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import warnings
warnings.filterwarnings("ignore")

from data_processing import preprocess
from backtest import run_backtest, run_equal_weight_baseline, compute_nav, align_series
from config import (
    RISK_FREE_RATE, INITIAL_CAPITAL, RESULTS_DIR,
    ABLATION_CONFIGS, WEIGHT_VALUE, WEIGHT_MOMENTUM, WEIGHT_SENTIMENT
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(ret_series: pd.Series, name: str) -> dict:
    """Compute full performance metric suite for a return series."""
    r = ret_series.dropna()
    if len(r) == 0:
        return {"Strategy": name}
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan
    cum     = (1 + r).cumprod()
    dd      = (cum / cum.cummax() - 1)
    max_dd  = dd.min()
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    total   = cum.iloc[-1] - 1
    return {
        "Strategy":        name,
        "Ann. Return":     f"{ann_ret:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio":    f"{sharpe:.3f}",
        "Max Drawdown":    f"{max_dd:.2%}",
        "Calmar Ratio":    f"{calmar:.3f}",
        "Total Return":    f"{total:.2%}",
        "_sharpe":  sharpe,
        "_ann_ret": ann_ret,
        "_max_dd":  max_dd,
    }


def compute_yearly_returns(ret_series: pd.Series) -> pd.Series:
    """
    Break down returns by calendar year for regime analysis.

    Args:
        ret_series: Monthly return series indexed by period

    Returns:
        Series of annual returns indexed by year
    """
    df = ret_series.reset_index()
    df.columns = ["ym", "return"]
    df["year"] = df["ym"].dt.year
    return df.groupby("year")["return"].apply(lambda x: (1 + x).prod() - 1)


def print_results_table(metrics_list: list) -> None:
    """Print formatted performance summary table to terminal."""
    cols = ["Strategy", "Ann. Return", "Ann. Volatility",
            "Sharpe Ratio", "Max Drawdown", "Calmar Ratio", "Total Return"]
    header = f"{'Strategy':<22}" + "".join(f"{c:>16}" for c in cols[1:])
    sep    = "=" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for m in metrics_list:
        row = f"{m.get('Strategy',''):<22}" + "".join(
            f"{m.get(c, 'N/A'):>16}" for c in cols[1:]
        )
        print(row)
    print(sep + "\n")


def print_regime_table(strategies: dict) -> None:
    """
    Print annual returns by regime for all strategies.

    Args:
        strategies: Dictionary of strategy name to return series
    """
    regime_labels = {
        2020: "COVID crash + recovery",
        2021: "Liquidity bull market",
        2022: "Inflationary bear market",
        2023: "AI-driven recovery",
    }
    print("\n" + "="*70)
    print("  REGIME SUBPERIOD ANALYSIS: Annual Returns by Year")
    print("="*70)
    header = f"{'Year':<6} {'Regime':<28}" + "".join(
        f"{k[:14]:>16}" for k in strategies.keys()
    )
    print(header)
    print("-"*70)
    for year in [2020, 2021, 2022, 2023]:
        row = f"{year:<6} {regime_labels.get(year,''):<28}"
        for ret_series in strategies.values():
            yearly = compute_yearly_returns(ret_series)
            val = yearly.get(year, np.nan)
            row += f"{val:>15.2%}" if not np.isnan(val) else f"{'N/A':>15}"
        print(row)
    print("="*70 + "\n")


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_cumulative_returns(nav_dict: dict) -> None:
    """Plot cumulative portfolio value for all strategies."""
    fig, ax = plt.subplots(figsize=(12, 6))
    styles = ["-", "--", "-.", ":", "-", "--", "-."]
    for i, (name, nav) in enumerate(nav_dict.items()):
        idx = [str(p) for p in nav.index]
        ax.plot(idx, nav.values, label=name, linewidth=2,
                linestyle=styles[i % len(styles)])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    step = max(1, len(idx) // 8)
    ax.set_xticks(range(0, len(idx), step))
    ax.set_xticklabels([idx[i] for i in range(0, len(idx), step)], rotation=45, ha="right")
    ax.set_title("Cumulative Portfolio Value, All Strategies", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "cumulative_returns.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved {path}")


def plot_drawdowns(ret_dict: dict) -> None:
    """Plot drawdown over time for all strategies."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, ret in ret_dict.items():
        cum = (1 + ret.fillna(0)).cumprod()
        dd  = (cum / cum.cummax() - 1)
        idx = [str(p) for p in dd.index]
        ax.plot(idx, dd.values * 100, label=name, linewidth=1.5)
    step = max(1, len(idx) // 8)
    ax.set_xticks(range(0, len(idx), step))
    ax.set_xticklabels([idx[i] for i in range(0, len(idx), step)], rotation=45, ha="right")
    ax.set_title("Drawdown Over Time, All Strategies", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "drawdown_chart.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved {path}")


def plot_ablation_bar(metrics_list: list) -> None:
    """Plot Sharpe ratio bar chart for ablation configurations."""
    ablation_names = list(ABLATION_CONFIGS.keys())
    ablation_metrics = [m for m in metrics_list if m["Strategy"] in ablation_names]
    names   = [m["Strategy"] for m in ablation_metrics]
    sharpes = [m["_sharpe"]  for m in ablation_metrics]
    colors  = ["#2ecc71" if n == "Full Model" else "#e74c3c" for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, sharpes, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Ablation Study, Sharpe Ratio by Model Configuration",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_ylim(0, max(sharpes) * 1.3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "ablation_bar_chart.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved {path}")


def plot_regime_bar(strategies: dict) -> None:
    """Plot annual returns by year for key strategies."""
    key_strategies = {k: v for k, v in strategies.items()
                     if k in ["Full Model", "Equal Weight", "Momentum Only"]}
    years = [2020, 2021, 2022, 2023]
    x = np.arange(len(years))
    width = 0.25
    colors = ["#2E86AB", "#A23B72", "#F18F01"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, ret) in enumerate(key_strategies.items()):
        yearly = compute_yearly_returns(ret)
        vals   = [yearly.get(y, 0) for y in years]
        bars   = ax.bar(x + i*width, [v*100 for v in vals],
                       width, label=name, color=colors[i], edgecolor="black", alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.5 if val >= 0 else -2.5),
                    f"{val:.1%}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width)
    regime_labels = ["2020\nCOVID crash", "2021\nBull market",
                     "2022\nBear market", "2023\nRecovery"]
    ax.set_xticklabels(regime_labels)
    ax.set_title("Annual Returns by Market Regime", fontsize=13, fontweight="bold")
    ax.set_ylabel("Annual Return (%)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "regime_analysis.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation() -> None:
    """Run full evaluation pipeline."""
    print("\n" + "="*60)
    print("  HYBRID QUANTAMENTAL MULTI-AGENT INVESTMENT SYSTEM")
    print("  Evaluation Report")
    print("  FIN 580 | Arielle Ward & Qiaoqi Li | May 2026")
    print("="*60 + "\n")

    print("Step 1: Loading data...")
    data = preprocess()

    print("Step 2: Running all strategies...")
    strategies = {}
    strategies["Full Model"]    = run_backtest(data, WEIGHT_VALUE, WEIGHT_MOMENTUM, WEIGHT_SENTIMENT)
    strategies["No Narrative"]  = run_backtest(data, 0.50, 0.50, 0.00)
    strategies["No Value"]      = run_backtest(data, 0.00, 0.50, 0.50)
    strategies["No Momentum"]   = run_backtest(data, 0.50, 0.00, 0.50)
    strategies["Equal Weight"]  = run_equal_weight_baseline(data)
    strategies["Momentum Only"] = run_backtest(data, 0.0, 1.0, 0.0)
    strategies["Value Only"]    = run_backtest(data, 1.0, 0.0, 0.0)
    strategies = align_series(strategies)

    print("Step 3: Computing metrics...")
    metrics_list = [compute_metrics(v, k) for k, v in strategies.items()]

    print("\nStep 4: Results\n")
    print_results_table(metrics_list)
    print_regime_table(strategies)

    print("Step 5: Generating charts...")
    nav_dict = {k: compute_nav(v) for k, v in strategies.items()}
    plot_cumulative_returns(nav_dict)
    plot_drawdowns(strategies)
    plot_ablation_bar(metrics_list)
    plot_regime_bar(strategies)

    print(f"\nAll results saved to /{RESULTS_DIR}/")
    print("Done.\n")


if __name__ == "__main__":
    run_evaluation()
