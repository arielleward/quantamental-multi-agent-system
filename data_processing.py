"""
data_processing.py
Data loading, cleaning, and preprocessing pipeline for the
Hybrid Quantamental Multi-Agent Investment System.

FIN 580 | Arielle Ward & Qiaoqi Li | May 2026

Usage:
    python data_processing.py

Outputs:
    - Prints dataset summary to terminal
    - Returns a clean, preprocessed DataFrame ready for agent processing
"""

import pandas as pd
import numpy as np
import os
from config import (
    DATA_FILE, START_DATE, END_DATE,
    MOMENTUM_LOOKBACK, SENTIMENT_SHIFT
)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw dataset from CSV file.

    Args:
        filepath: Path to the CSV file

    Returns:
        Raw DataFrame with parsed dates
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Make sure {DATA_FILE} is in the same directory."
        )
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df


def fill_missing_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill and back-fill missing fundamental values within each ticker.
    Only uses past data (forward-fill) to prevent look-ahead bias.
    Back-fill only used for initial NaN values at the start of the series.

    Args:
        df: Raw DataFrame

    Returns:
        DataFrame with filled fundamental columns
    """
    fundamental_cols = ["PE", "ROE", "PB"]
    df[fundamental_cols] = (
        df.groupby("ticker")[fundamental_cols]
        .ffill()
    )
    df[fundamental_cols] = (
        df.groupby("ticker")[fundamental_cols]
        .bfill()
    )
    return df


def fill_missing_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill missing momentum values within each ticker.

    Args:
        df: DataFrame with momentum column

    Returns:
        DataFrame with filled momentum column
    """
    df["momentum"] = df.groupby("ticker")["momentum"].ffill()
    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate the dataset and print a summary report.

    Args:
        df: Preprocessed DataFrame
    """
    print("\n" + "="*60)
    print("  DATA VALIDATION REPORT")
    print("="*60)
    print(f"  Total rows:       {len(df):,}")
    print(f"  Unique tickers:   {df['ticker'].nunique()}")
    print(f"  Date range:       {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Trading days:     {df['date'].nunique():,}")
    print("\n  Missing values after preprocessing:")
    for col in ["return", "momentum", "sentiment", "PE", "ROE", "PB"]:
        n_missing = df[col].isna().sum()
        pct = n_missing / len(df) * 100
        status = "OK" if pct < 5 else "WARNING"
        print(f"    {col:<12}: {n_missing:>5} missing ({pct:.1f}%) [{status}]")
    print("\n  Tickers in universe:")
    tickers = sorted(df["ticker"].unique())
    for i in range(0, len(tickers), 6):
        print("    " + "  ".join(tickers[i:i+6]))
    print("="*60 + "\n")


def filter_date_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to the configured date range.

    Args:
        df: Full DataFrame

    Returns:
        Filtered DataFrame
    """
    return df[
        (df["date"] >= START_DATE) &
        (df["date"] <= END_DATE)
    ].reset_index(drop=True)


def add_period_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a year-month period column for monthly aggregation.

    Args:
        df: DataFrame with date column

    Returns:
        DataFrame with ym (year-month period) column added
    """
    df["ym"] = df["date"].dt.to_period("M")
    return df


def compute_monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compound daily returns to monthly returns per ticker.
    This is the return series used by the backtesting engine.

    Temporal discipline: uses only returns within each month,
    no forward-looking data.

    Args:
        df: DataFrame with daily returns and ym column

    Returns:
        DataFrame with columns: ym, ticker, monthly_return
    """
    monthly = (
        df.groupby(["ym", "ticker"])["return"]
        .apply(lambda x: (1 + x.fillna(0)).prod() - 1)
        .reset_index()
    )
    monthly.columns = ["ym", "ticker", "monthly_return"]
    return monthly


def compute_monthly_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily sentiment scores to monthly mean per ticker.
    Sentiment data is already shifted forward by SENTIMENT_SHIFT days
    in the raw dataset to prevent look-ahead bias.

    Args:
        df: DataFrame with daily sentiment and ym column

    Returns:
        DataFrame with columns: ym, ticker, sentiment
    """
    return (
        df.groupby(["ym", "ticker"])["sentiment"]
        .mean()
        .reset_index()
    )


def get_rebalancing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the last trading day of each month for use as signal formation dates.
    Signals are formed on these dates and applied to the following month.

    Args:
        df: DataFrame with date and ym columns

    Returns:
        DataFrame with columns: ym, date (last trading day of each month)
    """
    return df.groupby("ym")["date"].max().reset_index()


def preprocess(filepath: str = DATA_FILE) -> dict:
    """
    Full preprocessing pipeline. Loads, cleans, validates, and
    computes all derived data structures needed by the agent system.

    Args:
        filepath: Path to the raw CSV file

    Returns:
        Dictionary containing:
            - df: Clean daily DataFrame
            - monthly_returns: Monthly compounded returns per ticker
            - monthly_sentiment: Monthly mean sentiment per ticker
            - rebal_dates: Last trading day of each month
    """
    print("Loading data...")
    df = load_raw_data(filepath)

    print("Filling missing values...")
    df = fill_missing_fundamentals(df)
    df = fill_missing_momentum(df)
    df = filter_date_range(df)
    df = add_period_column(df)

    validate_data(df)

    print("Computing monthly aggregates...")
    monthly_returns   = compute_monthly_returns(df)
    monthly_sentiment = compute_monthly_sentiment(df)
    rebal_dates       = get_rebalancing_dates(df)

    print(f"  Monthly return periods:    {monthly_returns['ym'].nunique()}")
    print(f"  Rebalancing dates:         {len(rebal_dates)}")
    print("Preprocessing complete.\n")

    return {
        "df": df,
        "monthly_returns": monthly_returns,
        "monthly_sentiment": monthly_sentiment,
        "rebal_dates": rebal_dates,
    }


if __name__ == "__main__":
    data = preprocess()
    print("Data is ready for agent processing.")
    print(f"Daily data shape: {data['df'].shape}")
    print(f"Monthly returns shape: {data['monthly_returns'].shape}")
