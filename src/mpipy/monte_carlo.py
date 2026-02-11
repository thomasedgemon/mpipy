"""Parallel Monte Carlo utilities with customizable reducers."""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .config import get_config
from .runtime import COMM_WORLD, LocalComm, cancel_requested, run


@dataclass(frozen=True)
class MonteCarloResult:
    mean: float
    variance: float
    stderr: float
    samples: int


def _partition_counts(total: int, parts: int) -> list[int]:
    base = total // parts
    remainder = total % parts
    return [base + (1 if i < remainder else 0) for i in range(parts)]


def _default_init() -> dict[str, float]:
    return {"sum": 0.0, "sumsq": 0.0, "count": 0.0}


def _default_reduce(acc: dict[str, float], value: float) -> dict[str, float]:
    acc["sum"] += value
    acc["sumsq"] += value * value
    acc["count"] += 1.0
    return acc


def _default_combine(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    left["sum"] += right["sum"]
    left["sumsq"] += right["sumsq"]
    left["count"] += right["count"]
    return left


def _default_finalize(acc: dict[str, float], total_samples: int) -> MonteCarloResult:
    if total_samples <= 0:
        return MonteCarloResult(mean=float("nan"), variance=float("nan"), stderr=float("nan"), samples=0)
    mean = acc["sum"] / total_samples
    variance = max(0.0, acc["sumsq"] / total_samples - mean * mean)
    stderr = math.sqrt(variance / total_samples)
    return MonteCarloResult(mean=mean, variance=variance, stderr=stderr, samples=total_samples)


def _monte_carlo_impl(
    num_samples: int,
    sample_fn: Callable[[random.Random], Any],
    eval_fn: Callable[[Any], float],
    init_fn: Optional[Callable[[], Any]],
    reduce_fn: Optional[Callable[[Any, Any], Any]],
    combine_fn: Optional[Callable[[Any, Any], Any]],
    finalize_fn: Optional[Callable[[Any, int], Any]],
    seed: Optional[int],
    cancel_check_every: int,
    comm,
):
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if reduce_fn is None:
        init_fn = _default_init
        reduce_fn = _default_reduce
        combine_fn = _default_combine
        finalize_fn = _default_finalize
    else:
        if init_fn is None:
            raise ValueError("init_fn is required when reduce_fn is provided")
        if combine_fn is None:
            raise ValueError("combine_fn is required when reduce_fn is provided")

    counts = _partition_counts(num_samples, comm.size)
    local_samples = counts[comm.rank]
    rng = random.Random(seed + comm.rank if seed is not None else None)

    acc = init_fn()
    cancelled = False
    for i in range(local_samples):
        if cancel_check_every and i % cancel_check_every == 0 and cancel_requested():
            cancelled = True
            break
        sample = sample_fn(rng)
        value = eval_fn(sample)
        acc = reduce_fn(acc, value)

    partials = comm.gather((cancelled, acc), root=0)
    if comm.rank != 0:
        return None

    if any(flag for flag, _ in partials):
        return None

    combined = partials[0][1]
    for _, partial in partials[1:]:
        combined = combine_fn(combined, partial)

    if finalize_fn is None:
        return combined
    return finalize_fn(combined, num_samples)


def _monte_carlo_entry(
    num_samples: int,
    sample_fn: Callable[[random.Random], Any],
    eval_fn: Callable[[Any], float],
    init_fn: Optional[Callable[[], Any]],
    reduce_fn: Optional[Callable[[Any, Any], Any]],
    combine_fn: Optional[Callable[[Any, Any], Any]],
    finalize_fn: Optional[Callable[[Any, int], Any]],
    seed: Optional[int],
    cancel_check_every: int,
):
    comm = COMM_WORLD or LocalComm()
    return _monte_carlo_impl(
        num_samples,
        sample_fn,
        eval_fn,
        init_fn,
        reduce_fn,
        combine_fn,
        finalize_fn,
        seed,
        cancel_check_every,
        comm,
    )


def monte_carlo(
    num_samples: int,
    sample_fn: Callable[[random.Random], Any],
    eval_fn: Callable[[Any], float],
    *,
    init_fn: Optional[Callable[[], Any]] = None,
    reduce_fn: Optional[Callable[[Any, Any], Any]] = None,
    combine_fn: Optional[Callable[[Any, Any], Any]] = None,
    finalize_fn: Optional[Callable[[Any, int], Any]] = None,
    seed: Optional[int] = None,
    cancel_check_every: int = 1024,
    comm: Optional[object] = None,
):
    """Run a Monte Carlo estimate across ranks.

    sample_fn must accept a random.Random instance and return a sample.
    eval_fn should map a sample to a numeric value (for the default reducer).

    If reduce_fn is provided, init_fn and combine_fn are required. finalize_fn
    is optional; if omitted, the combined accumulator is returned on rank 0.
    """

    if comm is None:
        comm = COMM_WORLD
    if comm is None:
        cfg = get_config()
        if cfg is not None and os.environ.get("MPI_RANK") is None:
            return run(
                _monte_carlo_entry,
                num_samples,
                sample_fn,
                eval_fn,
                init_fn,
                reduce_fn,
                combine_fn,
                finalize_fn,
                seed,
                cancel_check_every,
            )
        comm = LocalComm()
    return _monte_carlo_impl(
        num_samples,
        sample_fn,
        eval_fn,
        init_fn,
        reduce_fn,
        combine_fn,
        finalize_fn,
        seed,
        cancel_check_every,
        comm,
    )
