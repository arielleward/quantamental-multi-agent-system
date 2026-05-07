# ============================================================
# config.py
# Configuration file for the Hybrid Quantamental
# Multi-Agent Investment System
# FIN 580 | Arielle Ward & Qiaoqi Li | May 2026
# ============================================================

# ── Data ─────────────────────────────────────────────────────
DATA_FILE = "final_dataset.csv"
RESULTS_DIR = "results"

# ── Universe ──────────────────────────────────────────────────
TICKERS = [
    "AAPL", "AMD", "AMZN", "BAC", "COST", "CSCO", "CVX", "DIS",
    "GOOGL", "GS", "HD", "INTC", "JNJ", "JPM", "KO", "MA", "MCD",
    "META", "MSFT", "NFLX", "NKE", "NVDA", "PEP", "PFE", "PG",
    "TSLA", "UNH", "V", "WMT", "XOM"
]

START_DATE = "2020-01-02"
END_DATE   = "2023-12-29"

# ── Portfolio ─────────────────────────────────────────────────
INITIAL_CAPITAL  = 1_000_000   # starting capital in USD
TRANSACTION_COST = 0.003       # 30bps round-trip per rebalance
TOP_N_STOCKS     = 10          # number of stocks selected each month
RISK_FREE_RATE   = 0.02        # annual risk-free rate for Sharpe calculation

# ── Agent Weights ─────────────────────────────────────────────
# Must sum to 1.0
WEIGHT_VALUE     = 0.30        # Value Agent weight
WEIGHT_MOMENTUM  = 0.30        # Momentum Agent weight
WEIGHT_SENTIMENT = 0.40        # Verified Sentiment weight

# ── Agent Parameters ─────────────────────────────────────────
MOMENTUM_LOOKBACK   = 126      # trading days for momentum signal (6 months)
SENTIMENT_SHIFT     = 1        # days to shift sentiment forward (prevent leakage)

# Verification Agent thresholds
VERIFICATION_HIGH_SENTIMENT = 0.60   # sentiment above this is "strongly positive"
VERIFICATION_LOW_VALUE      = 0.30   # value below this is "weak fundamentals"
VERIFICATION_LOW_SENTIMENT  = 0.40   # sentiment below this is "negative"
VERIFICATION_HIGH_VALUE     = 0.70   # value above this is "strong fundamentals"
VERIFICATION_PENALTY        = 0.50   # multiply sentiment by this when penalizing
VERIFICATION_BOOST          = 1.30   # multiply sentiment by this when boosting
VERIFICATION_CAP            = 1.00   # maximum verified sentiment score

# ── Ablation Configurations ───────────────────────────────────
ABLATION_CONFIGS = {
    "Full Model":   {"w_val": 0.30, "w_mom": 0.30, "w_sent": 0.40},
    "No Narrative": {"w_val": 0.50, "w_mom": 0.50, "w_sent": 0.00},
    "No Value":     {"w_val": 0.00, "w_mom": 0.50, "w_sent": 0.50},
    "No Momentum":  {"w_val": 0.50, "w_mom": 0.00, "w_sent": 0.50},
}

# ── Baseline Configurations ───────────────────────────────────
BASELINE_CONFIGS = {
    "Equal Weight":   "equal_weight",
    "Momentum Only":  {"w_val": 0.00, "w_mom": 1.00, "w_sent": 0.00},
    "Value Only":     {"w_val": 1.00, "w_mom": 0.00, "w_sent": 0.00},
}
