"""Strategy parameter types and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


class ParameterValidationError(ValueError):
    """Raised when strategy parameters fail validation."""


@dataclass(frozen=True)
class MovingAverageParams:
    """Validated moving-average crossover periods."""

    fast_period: int
    slow_period: int

    def __post_init__(self) -> None:
        validate_ma_periods(self.fast_period, self.slow_period)

    @property
    def label(self) -> str:
        return f"{self.fast_period}/{self.slow_period}"

    def to_dict(self) -> dict[str, int]:
        return {"fast_period": self.fast_period, "slow_period": self.slow_period}

    def to_backtrader_kwargs(self) -> dict[str, int]:
        return self.to_dict()


def validate_ma_periods(fast_period: int, slow_period: int) -> None:
    """Ensure moving-average periods are valid.

    Rules:
        - both periods > 1
        - fast_period < slow_period

    Raises:
        ParameterValidationError: If validation fails.
    """
    if fast_period <= 1:
        raise ParameterValidationError(
            f"fast_period must be greater than 1 (got {fast_period})"
        )
    if slow_period <= 1:
        raise ParameterValidationError(
            f"slow_period must be greater than 1 (got {slow_period})"
        )
    if fast_period >= slow_period:
        raise ParameterValidationError(
            f"fast_period ({fast_period}) must be less than slow_period ({slow_period})"
        )


def parse_ma_periods(
    fast_period: int | None,
    slow_period: int | None,
    *,
    default_fast: int = 20,
    default_slow: int = 50,
) -> MovingAverageParams:
    """Parse and validate MA periods, applying defaults for missing values."""
    fast = default_fast if fast_period is None else int(fast_period)
    slow = default_slow if slow_period is None else int(slow_period)
    return MovingAverageParams(fast_period=fast, slow_period=slow)


DEFAULT_MA_OPTIMIZATION_GRID: tuple[tuple[int, int], ...] = (
    (10, 30),
    (10, 50),
    (20, 50),
    (20, 100),
    (50, 200),
)


def ma_grid_from_strings(grid: Sequence[str] | None) -> tuple[tuple[int, int], ...]:
    """Parse 'fast/slow' strings into period tuples."""
    if not grid:
        return DEFAULT_MA_OPTIMIZATION_GRID
    parsed: list[tuple[int, int]] = []
    for item in grid:
        parts = item.replace(",", "/").split("/")
        if len(parts) != 2:
            raise ParameterValidationError(
                f"Invalid grid entry '{item}'; expected format fast/slow"
            )
        fast, slow = int(parts[0].strip()), int(parts[1].strip())
        validate_ma_periods(fast, slow)
        parsed.append((fast, slow))
    return tuple(parsed)
