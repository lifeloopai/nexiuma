"""Strategy registry for discovery and instantiation."""

from __future__ import annotations

from typing import Type

import backtrader as bt

from strategies.base_strategy import NexiumaStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.moving_average import MovingAverageCrossover
from strategies.rsi_strategy import RSIStrategy

STRATEGY_REGISTRY: dict[str, Type[NexiumaStrategy]] = {
    "moving_average": MovingAverageCrossover,
    "ma": MovingAverageCrossover,
    "rsi": RSIStrategy,
    "momentum": MomentumStrategy,
}


def get_strategy_class(name: str) -> Type[NexiumaStrategy]:
    """Resolve strategy class by name (case-insensitive).

    Raises:
        KeyError: If strategy name is unknown.
    """
    key = name.lower().strip()
    if key not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(set(STRATEGY_REGISTRY.keys())))
        raise KeyError(f"Unknown strategy '{name}'. Available: {available}")
    return STRATEGY_REGISTRY[key]


def list_strategies() -> list[dict[str, str]]:
    """Return metadata for all registered strategies."""
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for cls in STRATEGY_REGISTRY.values():
        if cls.strategy_name in seen:
            continue
        seen.add(cls.strategy_name)
        result.append(
            {
                "name": cls.strategy_name,
                "description": cls.strategy_description,
                "class": cls.__name__,
            }
        )
    return sorted(result, key=lambda x: x["name"])
