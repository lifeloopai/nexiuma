"""Risk analytics: VaR, CVaR, beta, and drawdown analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskMetrics:
    """Extended risk statistics beyond core backtest metrics."""

    var_95: float
    cvar_95: float
    max_drawdown_duration_days: int
    avg_drawdown: float
    calmar_ratio: float
    ulcer_index: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "avg_drawdown": self.avg_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "ulcer_index": self.ulcer_index,
        }


class RiskAnalyzer:
    """Compute portfolio risk metrics from returns or equity curve."""

    def __init__(self, confidence: float = 0.95) -> None:
        self.confidence = confidence

    def analyze(
        self,
        equity_curve: pd.Series,
        annualized_return: float,
        periods_per_year: int = 252,
    ) -> RiskMetrics:
        """Derive risk metrics from equity curve."""
        values = equity_curve.astype(float)
        returns = values.pct_change().dropna()

        var_95 = float(np.percentile(returns, (1 - self.confidence) * 100)) if len(returns) else 0.0
        tail = returns[returns <= var_95]
        cvar_95 = float(tail.mean()) if len(tail) else var_95

        cummax = values.cummax()
        drawdown = (values - cummax) / cummax
        max_dd = drawdown.min()
        avg_drawdown = float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0

        calmar = annualized_return / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0

        squared_dd = (drawdown ** 2).mean()
        ulcer = float(np.sqrt(squared_dd)) if squared_dd > 0 else 0.0

        dd_duration = self._max_drawdown_duration(drawdown)

        return RiskMetrics(
            var_95=round(var_95, 6),
            cvar_95=round(cvar_95, 6),
            max_drawdown_duration_days=dd_duration,
            avg_drawdown=round(avg_drawdown, 6),
            calmar_ratio=round(calmar, 4),
            ulcer_index=round(ulcer, 6),
        )

    @staticmethod
    def _max_drawdown_duration(drawdown: pd.Series) -> int:
        """Longest consecutive period in drawdown."""
        in_dd = drawdown < 0
        if not in_dd.any():
            return 0
        groups = (~in_dd).cumsum()
        durations = in_dd.groupby(groups).sum()
        return int(durations.max()) if len(durations) else 0
