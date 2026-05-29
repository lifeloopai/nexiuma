"""Central configuration module for Nexiuma.

Settings can be loaded from environment variables (.env) and overridden via CLI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

# Project root (Nexiuma/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _parse_date(value: str | None, default: str) -> date:
    raw = value or default
    return datetime.strptime(raw, "%Y-%m-%d").date()


@dataclass(frozen=True)
class RiskSettings:
    """Risk management parameters for backtests and live trading."""

    initial_capital: float = 100_000.0
    position_size_pct: float = 0.95
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    max_position_size_pct: float = 1.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    risk_free_rate: float = 0.04
    use_volatility_sizing: bool = False
    target_volatility: float = 0.15


@dataclass(frozen=True)
class StrategySettings:
    """Per-strategy parameters (e.g. moving-average periods)."""

    fast_period: int = 20
    slow_period: int = 50


@dataclass(frozen=True)
class DataSettings:
    """Market data download and cache configuration."""

    ticker: str = "AAPL"
    start_date: date = field(default_factory=lambda: date(2020, 1, 1))
    end_date: date = field(default_factory=lambda: date(2024, 12, 31))
    auto_refresh: bool = False
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "cache")


@dataclass(frozen=True)
class NexiumaSettings:
    """Top-level application settings."""

    strategy: str = "moving_average"
    strategy_params: StrategySettings = field(default_factory=StrategySettings)
    data: DataSettings = field(default_factory=DataSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    backtest_results_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "backtests" / "results"
    )
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports" / "output")
    comparisons_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "reports" / "comparisons"
    )
    optimization_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "reports" / "optimization"
    )
    universe_optimization_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "reports" / "universe_optimization"
    )
    walkforward_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "reports" / "walkforward"
    )
    walkforward_universe_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "reports" / "walkforward_universe"
    )
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")

    def ensure_directories(self) -> None:
        """Create required output directories if missing."""
        for path in (
            self.data.cache_dir,
            self.backtest_results_dir,
            self.reports_dir,
            self.comparisons_dir,
            self.optimization_dir,
            self.universe_optimization_dir,
            self.walkforward_dir,
            self.walkforward_universe_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


_settings: NexiumaSettings | None = None


def load_settings(env_file: Path | None = None, overrides: dict[str, Any] | None = None) -> NexiumaSettings:
    """Load settings from .env and optional overrides (e.g. CLI args).

    Args:
        env_file: Path to .env file. Defaults to PROJECT_ROOT / '.env'.
        overrides: Dict of setting keys to override after env load.

    Returns:
        Fully constructed NexiumaSettings instance.
    """
    global _settings

    env_path = env_file or PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.debug("Loaded environment from {}", env_path)
    else:
        load_dotenv(env_path, override=False)

    risk = RiskSettings(
        initial_capital=float(os.getenv("INITIAL_CAPITAL", "100000")),
        position_size_pct=float(os.getenv("POSITION_SIZE_PCT", "0.95")),
        stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "0.05")),
        take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "0.15")),
        max_position_size_pct=float(os.getenv("MAX_POSITION_SIZE_PCT", "1.0")),
        commission_pct=float(os.getenv("COMMISSION_PCT", "0.001")),
        slippage_pct=float(os.getenv("SLIPPAGE_PCT", "0.0005")),
        risk_free_rate=float(os.getenv("RISK_FREE_RATE", "0.04")),
        use_volatility_sizing=_parse_bool(os.getenv("USE_VOLATILITY_SIZING"), False),
        target_volatility=float(os.getenv("TARGET_VOLATILITY", "0.15")),
    )

    cache_dir = Path(os.getenv("DATA_CACHE_DIR", str(PROJECT_ROOT / "data" / "cache")))
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir

    data = DataSettings(
        ticker=os.getenv("TICKER", "AAPL").upper(),
        start_date=_parse_date(os.getenv("START_DATE"), "2020-01-01"),
        end_date=_parse_date(os.getenv("END_DATE"), "2024-12-31"),
        auto_refresh=_parse_bool(os.getenv("AUTO_REFRESH"), False),
        cache_dir=cache_dir,
    )

    def _resolve_path(key: str, default: Path) -> Path:
        raw = os.getenv(key)
        if raw is None:
            return default
        p = Path(raw)
        return p if p.is_absolute() else PROJECT_ROOT / p

    strategy_params = StrategySettings(
        fast_period=int(os.getenv("FAST_PERIOD", "20")),
        slow_period=int(os.getenv("SLOW_PERIOD", "50")),
    )

    settings = NexiumaSettings(
        strategy=os.getenv("STRATEGY", "moving_average").lower(),
        strategy_params=strategy_params,
        data=data,
        risk=risk,
        backtest_results_dir=_resolve_path(
            "BACKTEST_RESULTS_DIR", PROJECT_ROOT / "backtests" / "results"
        ),
        reports_dir=_resolve_path("REPORTS_DIR", PROJECT_ROOT / "reports" / "output"),
        comparisons_dir=_resolve_path(
            "COMPARISONS_DIR", PROJECT_ROOT / "reports" / "comparisons"
        ),
        optimization_dir=_resolve_path(
            "OPTIMIZATION_DIR", PROJECT_ROOT / "reports" / "optimization"
        ),
        universe_optimization_dir=_resolve_path(
            "UNIVERSE_OPTIMIZATION_DIR",
            PROJECT_ROOT / "reports" / "universe_optimization",
        ),
        walkforward_dir=_resolve_path(
            "WALKFORWARD_DIR", PROJECT_ROOT / "reports" / "walkforward"
        ),
        walkforward_universe_dir=_resolve_path(
            "WALKFORWARD_UNIVERSE_DIR",
            PROJECT_ROOT / "reports" / "walkforward_universe",
        ),
        logs_dir=_resolve_path("LOGS_DIR", PROJECT_ROOT / "logs"),
    )

    if overrides:
        settings = _apply_overrides(settings, overrides)

    from strategies.parameters import parse_ma_periods

    parse_ma_periods(
        settings.strategy_params.fast_period,
        settings.strategy_params.slow_period,
    )

    settings.ensure_directories()
    _settings = settings
    return settings


def _apply_overrides(settings: NexiumaSettings, overrides: dict[str, Any]) -> NexiumaSettings:
    """Apply CLI or programmatic overrides to settings."""
    data_dict = {
        "ticker": settings.data.ticker,
        "start_date": settings.data.start_date,
        "end_date": settings.data.end_date,
        "auto_refresh": settings.data.auto_refresh,
        "cache_dir": settings.data.cache_dir,
    }
    risk_dict = {
        "initial_capital": settings.risk.initial_capital,
        "position_size_pct": settings.risk.position_size_pct,
        "stop_loss_pct": settings.risk.stop_loss_pct,
        "take_profit_pct": settings.risk.take_profit_pct,
        "max_position_size_pct": settings.risk.max_position_size_pct,
        "commission_pct": settings.risk.commission_pct,
        "slippage_pct": settings.risk.slippage_pct,
        "risk_free_rate": settings.risk.risk_free_rate,
        "use_volatility_sizing": settings.risk.use_volatility_sizing,
        "target_volatility": settings.risk.target_volatility,
    }
    strategy = settings.strategy
    strategy_params_dict = {
        "fast_period": settings.strategy_params.fast_period,
        "slow_period": settings.strategy_params.slow_period,
    }
    backtest_results_dir = settings.backtest_results_dir
    reports_dir = settings.reports_dir
    comparisons_dir = settings.comparisons_dir
    optimization_dir = settings.optimization_dir
    universe_optimization_dir = settings.universe_optimization_dir
    walkforward_dir = settings.walkforward_dir
    walkforward_universe_dir = settings.walkforward_universe_dir

    for key, value in overrides.items():
        if value is None:
            continue
        if key == "ticker":
            data_dict["ticker"] = str(value).upper()
        elif key == "start_date":
            data_dict["start_date"] = value if isinstance(value, date) else _parse_date(str(value), "2020-01-01")
        elif key == "end_date":
            data_dict["end_date"] = value if isinstance(value, date) else _parse_date(str(value), "2024-12-31")
        elif key == "auto_refresh":
            data_dict["auto_refresh"] = bool(value)
        elif key in risk_dict:
            risk_dict[key] = float(value) if key != "risk_free_rate" else float(value)
        elif key == "strategy":
            strategy = str(value).lower()
        elif key == "fast_period":
            strategy_params_dict["fast_period"] = int(value)
        elif key == "slow_period":
            strategy_params_dict["slow_period"] = int(value)
        elif key == "backtest_results_dir":
            backtest_results_dir = Path(value)
        elif key == "reports_dir":
            reports_dir = Path(value)
        elif key == "comparisons_dir":
            comparisons_dir = Path(value)
        elif key == "optimization_dir":
            optimization_dir = Path(value)
        elif key == "universe_optimization_dir":
            universe_optimization_dir = Path(value)
        elif key == "walkforward_dir":
            walkforward_dir = Path(value)
        elif key == "walkforward_universe_dir":
            walkforward_universe_dir = Path(value)

    return NexiumaSettings(
        strategy=strategy,
        strategy_params=StrategySettings(**strategy_params_dict),
        data=DataSettings(**data_dict),
        risk=RiskSettings(**risk_dict),
        backtest_results_dir=backtest_results_dir,
        reports_dir=reports_dir,
        comparisons_dir=comparisons_dir,
        optimization_dir=optimization_dir,
        universe_optimization_dir=universe_optimization_dir,
        walkforward_dir=walkforward_dir,
        walkforward_universe_dir=walkforward_universe_dir,
        logs_dir=settings.logs_dir,
    )


def get_settings() -> NexiumaSettings:
    """Return cached settings or load defaults."""
    global _settings
    if _settings is None:
        return load_settings()
    return _settings


def configure_logging(settings: NexiumaSettings | None = None) -> None:
    """Configure loguru to write to logs directory."""
    s = settings or get_settings()
    s.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = s.logs_dir / "nexiuma_{time:YYYY-MM-DD}.log"
    log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    logger.add(log_path, rotation="1 day", retention="30 days", level="INFO", format=log_format)
    logger.add(lambda msg: print(msg, end=""), level="INFO", format=log_format)
