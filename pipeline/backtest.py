"""Backtest harness (spec section 6). The implementation now lives in pipeline/simulate.py, which
generalises it into a parameterised simulator; this module stays as the import path the CLI and the
tests already use.

    from pipeline.backtest import run_backtest, score_gameweek
"""
from __future__ import annotations

from pipeline.simulate import (  # noqa: F401
    DEFAULT_HIT_RISK_PREMIUM as HIT_RISK_PREMIUM,
    START_BUDGET,
    BacktestResult,
    GWResult,
    run_backtest,
    score_gameweek,
)

__all__ = ["run_backtest", "score_gameweek", "BacktestResult", "GWResult", "START_BUDGET", "HIT_RISK_PREMIUM"]
