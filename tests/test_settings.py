"""Tests for configuration module."""

from __future__ import annotations

from datetime import date

from config.settings import RiskSettings, load_settings


def test_load_settings_defaults() -> None:
    settings = load_settings(overrides={"ticker": "MSFT", "strategy": "rsi"})
    assert settings.data.ticker == "MSFT"
    assert settings.strategy == "rsi"
    assert settings.risk.initial_capital == 100_000.0
    assert settings.comparisons_dir.name == "comparisons"


def test_risk_settings_frozen() -> None:
    risk = RiskSettings(stop_loss_pct=0.03)
    assert risk.stop_loss_pct == 0.03


def test_date_override() -> None:
    settings = load_settings(
        overrides={
            "start_date": date(2019, 1, 1),
            "end_date": date(2023, 12, 31),
        }
    )
    assert settings.data.start_date == date(2019, 1, 1)
    assert settings.data.end_date == date(2023, 12, 31)
