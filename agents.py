"""
agents.py
Agent logic and workflow for the Hybrid Quantamental
Multi-Agent Investment System.

Five agents, each with a distinct analytical function:
    1. ValueAgent         - fundamental valuation scoring
    2. MomentumAgent      - price trend scoring
    3. NarrativeAgent     - news sentiment scoring
    4. VerificationAgent  - cross-signal consistency checking
    5. PortfolioManager   - signal aggregation and stock selection

FIN 580 | Arielle Ward & Qiaoqi Li | May 2026
"""

import pandas as pd
import numpy as np
from config import (
    WEIGHT_VALUE, WEIGHT_MOMENTUM, WEIGHT_SENTIMENT,
    VERIFICATION_HIGH_SENTIMENT, VERIFICATION_LOW_VALUE,
    VERIFICATION_LOW_SENTIMENT, VERIFICATION_HIGH_VALUE,
    VERIFICATION_PENALTY, VERIFICATION_BOOST, VERIFICATION_CAP,
    TOP_N_STOCKS
)


# ── Utility ───────────────────────────────────────────────────────────────────

def cross_sectional_rank(series: pd.Series) -> pd.Series:
    """
    Compute cross-sectional percentile rank scaled to [0, 1].
    Rank of 1.0 = highest value in universe, 0.0 = lowest.

    Args:
        series: Series of signal values across tickers

    Returns:
        Series of percentile ranks in [0, 1]
    """
    return series.rank(pct=True)


# ── Agent 1: Value Agent ──────────────────────────────────────────────────────

class ValueAgent:
    """
    Information Gathering and Signal Generation agent for fundamental value.

    Evaluates each stock's fundamental attractiveness using three metrics:
    - Price-to-Earnings (PE): lower is better
    - Return on Equity (ROE): higher is better
    - Price-to-Book (PB): lower is better

    Each metric is ranked cross-sectionally and averaged to produce
    a composite value_score in [0, 1]. Treating each ratio as a
    separate sub-signal preserves diagnostic information and reflects
    the fine-grained task decomposition principle of Miyazaki et al. (2026).
    """

    def __init__(self):
        self.name = "Value Agent"
        self.role = "Information Gathering + Signal Generation"

    def score(self, snapshot: pd.DataFrame) -> pd.Series:
        """
        Compute value scores for all tickers at a given rebalancing date.

        Args:
            snapshot: DataFrame indexed by ticker with columns PE, ROE, PB

        Returns:
            Series of value_score values indexed by ticker
        """
        ranks = pd.DataFrame(index=snapshot.index)
        ranks["pe_rank"]  = cross_sectional_rank(-snapshot["PE"])   # lower PE = better
        ranks["roe_rank"] = cross_sectional_rank(snapshot["ROE"])   # higher ROE = better
        ranks["pb_rank"]  = cross_sectional_rank(-snapshot["PB"])   # lower PB = better
        return ranks.mean(axis=1).rename("value_score")

    def run(self, snapshot: pd.DataFrame) -> pd.Series:
        """Main entry point for the Value Agent."""
        return self.score(snapshot)


# ── Agent 2: Momentum Agent ───────────────────────────────────────────────────

class MomentumAgent:
    """
    Signal Generation agent for price momentum.

    Ranks stocks by their prior 126-day (6-month) trailing return
    cross-sectionally, producing a momentum_score in [0, 1].

    The 6-month horizon captures the bulk of the momentum premium
    while avoiding short-term reversal at 1-month horizons.
    Grounded in Baltussen et al. (2026): momentum is confirmed
    across 150 years and 46 countries as one of the most durable
    return predictors in finance.
    """

    def __init__(self):
        self.name = "Momentum Agent"
        self.role = "Signal Generation"

    def score(self, snapshot: pd.DataFrame) -> pd.Series:
        """
        Compute momentum scores for all tickers at a given rebalancing date.

        Args:
            snapshot: DataFrame indexed by ticker with momentum column

        Returns:
            Series of momentum_score values indexed by ticker
        """
        return cross_sectional_rank(snapshot["momentum"]).rename("momentum_score")

    def run(self, snapshot: pd.DataFrame) -> pd.Series:
        """Main entry point for the Momentum Agent."""
        return self.score(snapshot)


# ── Agent 3: Narrative Agent ──────────────────────────────────────────────────

class NarrativeAgent:
    """
    Interpretation agent for financial news sentiment.

    Converts monthly aggregated news sentiment scores into a
    cross-sectionally ranked signal. Monthly aggregation reduces
    noise while preserving directional signal. The one-day shift
    applied during data preprocessing ensures no contemporaneous
    news contaminates portfolio decisions.

    Grounded in Kirtac & Germano (2024): LLM-derived sentiment
    achieves Sharpe ratios of 3.05 vs 1.23 for dictionary methods
    across 965,000 U.S. financial news articles.
    """

    def __init__(self):
        self.name = "Narrative Agent"
        self.role = "Interpretation + Signal Generation"

    def score(self, monthly_sentiment: pd.Series) -> pd.Series:
        """
        Compute sentiment scores from monthly aggregated sentiment values.

        Args:
            monthly_sentiment: Series of mean monthly sentiment per ticker

        Returns:
            Series of sentiment_score values indexed by ticker
        """
        return cross_sectional_rank(monthly_sentiment).rename("sentiment_score")

    def run(self, monthly_sentiment: pd.Series) -> pd.Series:
        """Main entry point for the Narrative Agent."""
        return self.score(monthly_sentiment)


# ── Agent 4: Verification Agent ───────────────────────────────────────────────

class VerificationAgent:
    """
    Risk Management agent for cross-signal consistency checking.

    This is the primary coordination mechanism in the system,
    directly implementing the Coordination Primacy Hypothesis
    of Nguyen & Pham (2026). It cross-checks the Narrative Agent's
    sentiment signal against the Value Agent's fundamental assessment,
    applying corrections where inconsistency suggests the narrative
    may be misleading.

    Two correction rules:
    1. Penalty: If sentiment is strongly positive (> HIGH_SENTIMENT threshold)
       but fundamentals are weak (< LOW_VALUE threshold), the narrative may
       be reflecting a sentiment bubble. Sentiment is penalized by PENALTY factor.

    2. Boost: If sentiment is negative (< LOW_SENTIMENT threshold) but
       fundamentals are strong (> HIGH_VALUE threshold), the narrative may
       be overreacting to short-term news. Sentiment is boosted by BOOST factor,
       capped at VERIFICATION_CAP.

    The result is a verified_sentiment_score that carries less risk of
    distorting the value-momentum core of the portfolio.
    """

    def __init__(self):
        self.name = "Verification Agent"
        self.role = "Risk Management + Coordination"

    def verify(self, scores: pd.DataFrame) -> pd.Series:
        """
        Apply verification corrections to sentiment scores.

        Args:
            scores: DataFrame with columns value_score and sentiment_score,
                    indexed by ticker

        Returns:
            Series of verified_sentiment_score values indexed by ticker
        """
        verified = scores["sentiment_score"].copy()

        # Rule 1: Penalty for overconfident positive sentiment
        mask_penalty = (
            (scores["sentiment_score"] > VERIFICATION_HIGH_SENTIMENT) &
            (scores["value_score"] < VERIFICATION_LOW_VALUE)
        )
        verified[mask_penalty] *= VERIFICATION_PENALTY

        # Rule 2: Boost for excessive pessimism in strong fundamentals
        mask_boost = (
            (scores["sentiment_score"] < VERIFICATION_LOW_SENTIMENT) &
            (scores["value_score"] > VERIFICATION_HIGH_VALUE)
        )
        verified[mask_boost] = (
            verified[mask_boost] * VERIFICATION_BOOST
        ).clip(upper=VERIFICATION_CAP)

        return verified.rename("verified_sentiment")

    def run(self, scores: pd.DataFrame) -> pd.Series:
        """Main entry point for the Verification Agent."""
        return self.verify(scores)

    def get_adjustments(self, scores: pd.DataFrame) -> pd.DataFrame:
        """
        Return a summary of which tickers were adjusted and why.
        Useful for interpretability and presentation demos.

        Args:
            scores: DataFrame with value_score and sentiment_score

        Returns:
            DataFrame showing tickers with adjustments applied
        """
        mask_penalty = (
            (scores["sentiment_score"] > VERIFICATION_HIGH_SENTIMENT) &
            (scores["value_score"] < VERIFICATION_LOW_VALUE)
        )
        mask_boost = (
            (scores["sentiment_score"] < VERIFICATION_LOW_SENTIMENT) &
            (scores["value_score"] > VERIFICATION_HIGH_VALUE)
        )
        adjustments = []
        for ticker in scores.index:
            if mask_penalty[ticker]:
                adjustments.append({
                    "ticker": ticker,
                    "action": "PENALTY",
                    "reason": "High sentiment + weak fundamentals",
                    "original": round(scores.loc[ticker, "sentiment_score"], 3),
                    "adjusted": round(scores.loc[ticker, "sentiment_score"] * VERIFICATION_PENALTY, 3),
                })
            elif mask_boost[ticker]:
                adjustments.append({
                    "ticker": ticker,
                    "action": "BOOST",
                    "reason": "Low sentiment + strong fundamentals",
                    "original": round(scores.loc[ticker, "sentiment_score"], 3),
                    "adjusted": min(
                        round(scores.loc[ticker, "sentiment_score"] * VERIFICATION_BOOST, 3),
                        VERIFICATION_CAP
                    ),
                })
        if adjustments:
            return pd.DataFrame(adjustments).set_index("ticker")
        return pd.DataFrame(columns=["action", "reason", "original", "adjusted"])


# ── Agent 5: Portfolio Manager ────────────────────────────────────────────────

class PortfolioManager:
    """
    Portfolio Construction and Final Trading Decisions agent.

    Aggregates outputs from all four signal agents into a single
    composite score, selects the top N stocks, and determines
    final portfolio weights.

    Aggregation formula:
        final_score = w_val * value_score
                    + w_mom * momentum_score
                    + w_sent * verified_sentiment_score

    The 40% weight on verified sentiment reflects the hypothesis
    that a properly verified narrative signal adds incremental value
    beyond the value-momentum core. This hypothesis is tested through
    the ablation study in evaluation.py.
    """

    def __init__(self, w_val=WEIGHT_VALUE, w_mom=WEIGHT_MOMENTUM,
                 w_sent=WEIGHT_SENTIMENT, top_n=TOP_N_STOCKS):
        self.name = "Portfolio Manager"
        self.role = "Portfolio Construction + Final Trading Decisions"
        self.w_val  = w_val
        self.w_mom  = w_mom
        self.w_sent = w_sent
        self.top_n  = top_n

    def aggregate(self, scores: pd.DataFrame) -> pd.Series:
        """
        Compute final composite scores for all tickers.

        Args:
            scores: DataFrame with columns value_score, momentum_score,
                    verified_sentiment, indexed by ticker

        Returns:
            Series of final_score values indexed by ticker
        """
        return (
            self.w_val  * scores["value_score"] +
            self.w_mom  * scores["momentum_score"] +
            self.w_sent * scores["verified_sentiment"]
        ).rename("final_score")

    def select_portfolio(self, scores: pd.DataFrame) -> pd.Index:
        """
        Select top N tickers by final score.

        Args:
            scores: DataFrame with final_score column

        Returns:
            Index of selected ticker symbols
        """
        return scores.nlargest(self.top_n, "final_score").index

    def run(self, scores: pd.DataFrame) -> tuple:
        """
        Main entry point for the Portfolio Manager.

        Args:
            scores: DataFrame with all agent scores

        Returns:
            Tuple of (selected_tickers Index, scores DataFrame with final_score)
        """
        scores = scores.copy()
        scores["final_score"] = self.aggregate(scores)
        selected = self.select_portfolio(scores)
        return selected, scores


# ── Multi-Agent Workflow ──────────────────────────────────────────────────────

class MultiAgentPipeline:
    """
    Orchestrates the full multi-agent workflow for a single rebalancing period.

    Workflow:
        1. Value Agent     -> value_score
        2. Momentum Agent  -> momentum_score
        3. Narrative Agent -> sentiment_score
        4. Verification Agent -> verified_sentiment (coordination step)
        5. Portfolio Manager  -> final_score, selected portfolio
    """

    def __init__(self, w_val=WEIGHT_VALUE, w_mom=WEIGHT_MOMENTUM,
                 w_sent=WEIGHT_SENTIMENT, top_n=TOP_N_STOCKS):
        self.value_agent       = ValueAgent()
        self.momentum_agent    = MomentumAgent()
        self.narrative_agent   = NarrativeAgent()
        self.verification_agent = VerificationAgent()
        self.portfolio_manager = PortfolioManager(w_val, w_mom, w_sent, top_n)

    def run(self, snapshot: pd.DataFrame,
            monthly_sentiment: pd.Series) -> tuple:
        """
        Run the full agent pipeline for one rebalancing period.

        Args:
            snapshot: DataFrame indexed by ticker with daily signal columns
                      (PE, ROE, PB, momentum) for the rebalancing date
            monthly_sentiment: Series of monthly mean sentiment per ticker

        Returns:
            Tuple of:
                - selected: Index of selected tickers
                - all_scores: DataFrame with all agent scores
        """
        # Align tickers across all data sources
        common_tickers = snapshot.index.intersection(monthly_sentiment.index)
        if len(common_tickers) < self.portfolio_manager.top_n:
            return pd.Index([]), pd.DataFrame()

        snapshot_aligned  = snapshot.loc[common_tickers]
        sentiment_aligned = monthly_sentiment.loc[common_tickers]

        # Run each agent
        value_scores    = self.value_agent.run(snapshot_aligned)
        momentum_scores = self.momentum_agent.run(snapshot_aligned)
        sentiment_scores = self.narrative_agent.run(sentiment_aligned)

        # Combine into score matrix
        all_scores = pd.concat(
            [value_scores, momentum_scores, sentiment_scores], axis=1
        ).dropna()

        if len(all_scores) < self.portfolio_manager.top_n:
            return pd.Index([]), pd.DataFrame()

        # Verification step (coordination mechanism)
        all_scores["verified_sentiment"] = self.verification_agent.run(all_scores)

        # Portfolio construction and final decisions
        selected, all_scores = self.portfolio_manager.run(all_scores)

        return selected, all_scores
