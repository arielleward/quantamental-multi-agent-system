"""
backtest.py
Backtesting engine for the Hybrid Quantamental
Multi-Agent Investment System.

Runs the full multi-agent pipeline across all rebalancing periods
and computes portfolio returns with transaction costs applied.

FIN 580 | Arielle Ward & Qiaoqi Li | May 2026

Usage:
    python backtest.py
"""

import pandas as pd
import numpy as np
from data_processing import preprocess
from agents import MultiAgentPipeline
from config import (
    INITIAL_CAPITAL, TRANSACTION_COST, TOP_N_STOCKS,
    WEIGHT_VALUE, WEIGHT_MOMENTUM, WEIGHT_SENTIMENT
)


def run_backtest(data: dict, w_val: float = WEIGHT_VALUE,
                 w_mom: float = WEIGHT_MOMENTUM,
                 w_sent: float = WEIGHT_SENTIMENT,
                 top_n: int = TOP_N_STOCKS,
                 trans_cost: float = TRANSACTION_COST) -> pd.Series:
    """
    Run the full multi-agent backtest across all rebalancing periods.

    Temporal discipline enforced:
    - Signals formed at month-end date T
    - Portfolio invested at start of month T+1
    - Returns measured over month T+1
    - Transaction costs deducted at each rebalancing event

    Args:
        data: Preprocessed data dictionary from data_processing.preprocess()
        w_val: Value Agent weight
        w_mom: Momentum Agent weight
        w_sent: Verified Sentiment weight
        top_n: Number of stocks to select each month
        trans_cost: Round-trip transaction cost per rebalance

    Returns:
        Series of monthly portfolio returns indexed by period
    """
    df               = data["df"]
    monthly_returns  = data["monthly_returns"]
    monthly_sent     = data["monthly_sentiment"]
    rebal_dates      = data["rebal_dates"]

    pipeline = MultiAgentPipeline(w_val, w_mom, w_sent, top_n)
    portfolio_returns = []

    for _, row in rebal_dates.iterrows():
        ym   = row["ym"]
        date = row["date"]

        # Get cross-sectional snapshot at rebalancing date
        snapshot = df[df["date"] == date].set_index("ticker")
        if len(snapshot) < top_n:
            continue

        # Get monthly sentiment for this period
        sent_this_month = monthly_sent[
            monthly_sent["ym"] == ym
        ].set_index("ticker")["sentiment"]

        if sent_this_month.empty:
            continue

        # Run full agent pipeline (signal T)
        selected, _ = pipeline.run(snapshot, sent_this_month)

        if len(selected) == 0:
            continue

        # Apply portfolio to next month's returns (return T+1)
        next_ym = ym + 1
        next_returns = monthly_returns[
            (monthly_returns["ym"] == next_ym) &
            (monthly_returns["ticker"].isin(selected))
        ]["monthly_return"]

        if next_returns.empty:
            continue

        # Equal weight, deduct transaction cost
        port_return = next_returns.mean() - trans_cost
        portfolio_returns.append({"ym": next_ym, "return": port_return})

    if not portfolio_returns:
        return pd.Series(dtype=float)

    return pd.DataFrame(portfolio_returns).set_index("ym")["return"]


def run_equal_weight_baseline(data: dict,
                               trans_cost: float = TRANSACTION_COST) -> pd.Series:
    """
    Run the equal weight baseline strategy.
    All tickers in universe get equal weight, monthly rebalanced.

    Args:
        data: Preprocessed data dictionary
        trans_cost: Round-trip transaction cost

    Returns:
        Series of monthly portfolio returns
    """
    monthly_returns = data["monthly_returns"]
    ew = (
        monthly_returns
        .groupby("ym")["monthly_return"]
        .mean() - trans_cost
    )
    return ew.rename("return")


def compute_nav(ret_series: pd.Series,
                initial_capital: float = INITIAL_CAPITAL) -> pd.Series:
    """
    Compute net asset value (NAV) from a monthly return series.

    Args:
        ret_series: Monthly return series
        initial_capital: Starting portfolio value in USD

    Returns:
        Series of portfolio values over time
    """
    return initial_capital * (1 + ret_series.fillna(0)).cumprod()


def align_series(series_dict: dict) -> dict:
    """
    Align all return series to a common index for comparison.

    Args:
        series_dict: Dictionary of strategy name to return series

    Returns:
        Dictionary with all series reindexed to common index
    """
    all_idx = sorted(set().union(*[s.index for s in series_dict.values()]))
    return {k: v.reindex(all_idx) for k, v in series_dict.items()}


if __name__ == "__main__":
    from config import ABLATION_CONFIGS

    print("Loading and preprocessing data...")
    data = preprocess()

    print("Running Full Model backtest...")
    full_model = run_backtest(data)
    nav = compute_nav(full_model)

    print(f"\nFull Model Summary:")
    print(f"  Periods:        {len(full_model)}")
    print(f"  Final NAV:      ${nav.iloc[-1]:,.0f}")
    print(f"  Total Return:   {(nav.iloc[-1] / nav.iloc[0] - 1):.2%}")
    print("\nRun evaluation.py for full metrics and ablation results.")
