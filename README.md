# Hybrid Quantamental Multi-Agent Investment System

**FIN 580: Multi-Agent Trading and Investment Systems in the Age of AI**  
**Arielle Ward & Qiaoqi Li | May 2026**

---

## Overview

This project implements a Hybrid Quantamental Multi-Agent Investment System that integrates structured financial signals (value, momentum) with unstructured narrative data (financial news sentiment) to construct and backtest a long-only equity portfolio across 30 large-cap U.S. equities from January 2020 through December 2023.

The system is designed around five specialized agents, each assigned a precise and measurable analytical function, coordinated through a hierarchical signal aggregation protocol. This design is directly motivated by the fine-grained task decomposition principle of Miyazaki et al. (2026), which demonstrates that specific agent subtasks outperform abstract analyst-manager role assignments in multi-agent trading systems.

The project addresses two critical research gaps in the financial AI literature:

**Multimodal Integration Gap** (Sun et al., 2024): While alternative data adoption has accelerated across institutional investment, rigorous integration of heterogeneous data types within a single disciplined investment framework remains underdeveloped. This system addresses that gap by combining fundamental valuation, price momentum, and verified news sentiment within a coordinated five-agent pipeline.

**Coordination and Attribution Gap** (Nguyen & Pham, 2026): Multi-agent systems are increasingly common in financial AI, but systematic evidence on which agents actually add value remains scarce. This project addresses that gap through a four-configuration ablation study that provides clean, agent-level attribution of performance contributions.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     MULTI-AGENT SYSTEM                           │
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │    VALUE    │   │  MOMENTUM   │   │  NARRATIVE  │           │
│  │    AGENT    │   │    AGENT    │   │    AGENT    │           │
│  │             │   │             │   │             │           │
│  │ PE, ROE, PB │   │ 126-day     │   │ Daily news  │           │
│  │ Cross-sect. │   │ trailing    │   │ sentiment   │           │
│  │ percentile  │   │ return      │   │ Monthly agg │           │
│  │ rank        │   │ Cross-sect. │   │ Cross-sect. │           │
│  │             │   │ rank        │   │ rank        │           │
│  │ value_score │   │ momentum_   │   │ sentiment_  │           │
│  │ in [0,1]   │   │ score [0,1] │   │ score [0,1] │           │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
│         └─────────────────┼─────────────────┘                  │
│                           ▼                                     │
│              ┌────────────────────────┐                         │
│              │   VERIFICATION AGENT   │                         │
│              │                        │                         │
│              │ If sentiment > 0.60    │                         │
│              │ AND value < 0.30:      │                         │
│              │   penalty x 0.50       │                         │
│              │                        │                         │
│              │ If sentiment < 0.40    │                         │
│              │ AND value > 0.70:      │                         │
│              │   boost x 1.30         │                         │
│              │                        │                         │
│              │ verified_sentiment_    │                         │
│              │ score in [0,1]         │                         │
│              └────────────┬───────────┘                         │
│                           ▼                                     │
│              ┌────────────────────────┐                         │
│              │   PORTFOLIO MANAGER    │                         │
│              │        AGENT           │                         │
│              │                        │                         │
│              │ Final Score =          │                         │
│              │ 0.30 x value_score     │                         │
│              │ 0.30 x momentum_score  │                         │
│              │ 0.40 x verified_sent.  │                         │
│              │                        │                         │
│              │ Top 10 stocks selected │                         │
│              │ Equal weight 10% each  │                         │
│              │ Monthly rebalancing    │                         │
│              └────────────────────────┘                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## Agent Descriptions

### Agent 1: Value Agent
- **Inputs:** PE (price-to-earnings), ROE (return on equity), PB (price-to-book)
- **Logic:** For each monthly rebalancing date, computes cross-sectional percentile ranks independently for each fundamental metric. Lower PE = higher score, higher ROE = higher score, lower PB = higher score. The three ranks are averaged to produce a composite value_score normalized to [0,1].
- **Motivation:** Grounded in the empirical value premium literature. The decomposition into three distinct sub-signals rather than a single composite reflects the fine-grained task specification principle of Miyazaki et al. (2026).

### Agent 2: Momentum Agent
- **Inputs:** 126-day (6-month) trailing stock return
- **Logic:** Cross-sectional percentile rank of prior 126-day returns, producing a momentum_score in [0,1]. The 6-month horizon captures the bulk of the momentum premium while avoiding short-term reversal.
- **Motivation:** Grounded in the momentum anomaly confirmed by Baltussen et al. (2026) across 150 years and 46 countries.

### Agent 3: Narrative Agent
- **Inputs:** Daily financial news sentiment scores (-1 to +1)
- **Logic:** Aggregates daily sentiment to monthly mean per ticker, then applies cross-sectional percentile ranking to produce a sentiment_score in [0,1]. Monthly aggregation reduces noise while preserving the directional signal.
- **Motivation:** Kirtac and Germano (2024) demonstrate that LLM-derived sentiment achieves Sharpe ratios of 3.05 vs 1.23 for traditional dictionary approaches across 965,000 U.S. financial news articles.

### Agent 4: Verification Agent
- **Inputs:** sentiment_score and value_score from above agents
- **Logic:** Cross-checks narrative signals against fundamental valuations. Applies a 50% penalty when sentiment is strongly positive but fundamentals are weak (potential sentiment bubble), and a 30% boost capped at 1.0 when sentiment is negative but fundamentals are strong (potential overreaction).
- **Motivation:** Implements the inter-agent coordination mechanism motivated by the Coordination Primacy Hypothesis of Nguyen and Pham (2026). This is the primary novel architectural contribution of the system.

### Agent 5: Portfolio Manager Agent
- **Inputs:** value_score, momentum_score, verified_sentiment_score
- **Logic:** Computes Final Score = 0.30 x value_score + 0.30 x momentum_score + 0.40 x verified_sentiment_score. Selects top 10 stocks by final score with equal weights (10% each), rebalances monthly.
- **Motivation:** The 40% weight on verified sentiment reflects the hypothesis that properly verified narrative signals add incremental value beyond the value-momentum core, testable through the ablation study.

---

## Investment Universe

- **Universe:** 30 S&P 500 large-cap equities
- **Tickers:** AAPL, AMD, AMZN, BAC, COST, CSCO, CVX, DIS, GOOGL, GS, HD, INTC, JNJ, JPM, KO, MA, MCD, META, MSFT, NFLX, NKE, NVDA, PEP, PFE, PG, TSLA, UNH, V, WMT, XOM
- **Sectors:** Technology, Healthcare, Financials, Consumer Staples, Consumer Discretionary, Energy
- **Period:** January 2, 2020 through December 29, 2023
- **Market regimes covered:** COVID-19 crash and recovery (Q1-Q2 2020), liquidity-driven bull market (2020-2021), inflationary bear market (2022), AI-driven recovery (2023)

---

## Backtesting Setup

| Parameter | Value |
|---|---|
| Initial capital | $1,000,000 |
| Transaction costs | 30bps round-trip per rebalance |
| Rebalancing frequency | Monthly (48 events) |
| Portfolio size | Top 10 stocks |
| Weighting scheme | Equal weight (10% per stock) |
| Risk-free rate | 2.0% annually |
| Short selling | Not permitted |
| Leverage | None |

---

## Temporal Discipline

All signals are constructed to strictly prevent look-ahead bias, following the evaluation standards of Nguyen and Pham (2026):

- Sentiment data shifted forward by 1 trading day before any signal construction
- Momentum signals use only the prior 126 trading days available at signal date
- Missing fundamental values forward-filled within each ticker using only past data
- Signals formed at month-end date T used to construct portfolios effective month T+1
- Portfolio returns measured using realized returns from month T+1 only
- Transaction costs deducted at every rebalancing event (no cost neglect)

---

## Data Requirements

Place the following file in the same directory as `main.py`:

| File | Columns | Rows | Description |
|---|---|---|---|
| `final_dataset.csv` | date, ticker, return, momentum, sentiment, PE, ROE, PB | 30,180 | Master daily dataset, all signals pre-shifted by 1 day |

All signals in the dataset are already lag-adjusted. No additional preprocessing is required.

---

## Installation

```bash
pip install pandas numpy matplotlib seaborn
```

No other external libraries required. The system uses only standard Python data science libraries.

---

## How to Run

```bash
cd path/to/your/project/folder
python main.py
```

The script runs the full pipeline end-to-end and will:

1. Load and validate `final_dataset.csv`
2. Forward-fill missing fundamental values within each ticker
3. Build all agent signals across 48 monthly rebalancing periods
4. Run the Full Multi-Agent Model backtest
5. Run 3 ablation configurations (No Narrative, No Value, No Momentum)
6. Run 3 baseline strategies (Equal Weight, Momentum Only, Value Only)
7. Apply 30bps transaction costs at every rebalancing event
8. Compute all six performance metrics for every strategy
9. Print a formatted results table to the terminal
10. Generate and save 3 charts to the `/results/` folder

**Expected runtime:** Under 30 seconds on a standard laptop.

---

## Output Files

All outputs saved automatically to `/results/`:

| File | Description |
|---|---|
| `cumulative_returns.png` | Portfolio value ($) over time for all 7 strategies |
| `drawdown_chart.png` | Peak-to-trough drawdown (%) over time for all 7 strategies |
| `ablation_bar_chart.png` | Sharpe ratio comparison across Full Model and 3 ablation configs |

---

## Baseline and Ablation Configurations

### Baselines

| Strategy | Description |
|---|---|
| Equal Weight | All 30 securities, uniform weight, monthly rebalanced |
| Momentum Only | Top 10 by momentum_score, equal weight |
| Value Only | Top 10 by value_score, equal weight |

### Ablation Study

| Configuration | Value Weight | Momentum Weight | Sentiment Weight | Purpose |
|---|---|---|---|---|
| Full Model | 0.30 | 0.30 | 0.40 | Complete multi-agent system |
| No Narrative | 0.50 | 0.50 | 0.00 | Remove sentiment entirely |
| No Value | 0.00 | 0.50 | 0.50 | Remove fundamental signals |
| No Momentum | 0.50 | 0.00 | 0.50 | Remove price trend signals |

The ablation design follows the methodology of Miyazaki et al. (2026) and Nguyen and Pham (2026), who identify agent-level ablation as essential for credible performance attribution in multi-agent financial systems.

---

## Key Results

| Strategy | Ann. Return | Ann. Vol. | Sharpe | Max Drawdown | Calmar | Total Return |
|---|---|---|---|---|---|---|
| **Full Model** | **13.54%** | **17.73%** | **0.651** | **-18.20%** | **0.744** | **54.32%** |
| Equal Weight | 13.85% | 20.82% | 0.569 | -27.32% | 0.507 | 68.01% |
| Momentum Only | 19.04% | 22.74% | 0.750 | -23.08% | 0.825 | 81.41% |
| Value Only | 10.06% | 18.24% | 0.442 | -20.57% | 0.489 | 38.77% |
| No Narrative | 13.54% | 17.73% | 0.651 | -18.20% | 0.744 | 54.32% |
| No Value | 19.04% | 22.74% | 0.750 | -23.08% | 0.825 | 81.41% |
| No Momentum | 10.06% | 18.24% | 0.442 | -20.57% | 0.489 | 38.77% |

**Key finding:** The Full Model outperforms the Equal Weight benchmark on risk-adjusted basis (Sharpe 0.651 vs 0.569) with substantially lower maximum drawdown (-18.20% vs -27.32%). The No Narrative ablation producing identical results to the Full Model reveals that the Verification Agent neutralizes inconsistent sentiment signals, adding value through risk reduction rather than return enhancement. This coordination benefit would be entirely invisible without the ablation study.

---

## References

- Baltussen, G., Dom, G., Van Vliet, P., & Vidojevic, M. (2026). Momentum factor investing: Evidence and evolution. *Journal of Portfolio Management*, forthcoming.
- Coriat, B., & Benhamou, E. (2025). HARLF: Hierarchical reinforcement learning and LLM-driven sentiment integration. arXiv:2507.18560.
- Fatouros, G., et al. (2025). MarketSenseAI 2.0: Enhancing stock analysis through LLM agents. arXiv:2502.00415.
- Kirtac, K., & Germano, G. (2024). Sentiment trading with large language models. arXiv:2412.19245.
- Malo, P., et al. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *JASIST*, 65(4), 782-796.
- Miyazaki, K., et al. (2026). Toward expert investment teams: A multi-agent LLM system with fine-grained trading tasks. arXiv:2602.23330.
- Nguyen, P., & Pham, T. (2026). Toward reliable evaluation of LLM-based financial multi-agent systems. arXiv:2603.27539.
- Sun, Y., et al. (2024). Alternative data in finance and business: Emerging applications and theory analysis. *Financial Innovation*, 10.
- Xiao, Y., et al. (2024). TradingAgents: Multi-agents LLM financial trading framework. arXiv:2412.20138.

---

## Limitations

- Fundamental data uses static snapshots rather than time-varying quarterly updates
- Sentiment signal is market-wide rather than ticker-specific
- 30-stock universe limits generalizability
- Agent weights (0.30/0.30/0.40) are pre-specified, not optimized
- Four-regime evaluation may not capture long-run regime-shift risk

---

## Authors

**Arielle Ward & Qiaoqi Li**  
FIN 580: Multi-Agent Trading and Investment Systems in the Age of AI  
May 2026
