"""Tests for strategy registry."""

from __future__ import annotations

import pytest

from strategies.moving_average import MovingAverageCrossover
from strategies.registry import get_strategy_class, list_strategies


def test_get_strategy_moving_average() -> None:
    cls = get_strategy_class("moving_average")
    assert cls is MovingAverageCrossover


def test_get_strategy_alias() -> None:
    cls = get_strategy_class("ma")
    assert cls is MovingAverageCrossover


def test_unknown_strategy_raises() -> None:
    with pytest.raises(KeyError, match="Unknown strategy"):
        get_strategy_class("nonexistent")


def test_list_strategies_not_empty() -> None:
    strategies = list_strategies()
    assert len(strategies) >= 3
    names = {s["name"] for s in strategies}
    assert "moving_average" in names
    assert "rsi" in names
    assert "momentum" in names
