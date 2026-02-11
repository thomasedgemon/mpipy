import random

from mpipy.monte_carlo import MonteCarloResult, monte_carlo


def sample_uniform(rng: random.Random) -> float:
    return rng.random()


def identity(x: float) -> float:
    return x


def test_monte_carlo_default_stats():
    result = monte_carlo(20_000, sample_uniform, identity, seed=12345)
    assert isinstance(result, MonteCarloResult)
    assert abs(result.mean - 0.5) < 0.02
    assert result.samples == 20_000


def test_monte_carlo_custom_reducer():
    def init():
        return 0.0

    def reduce_sum(acc: float, value: float) -> float:
        return acc + value

    def combine_sum(left: float, right: float) -> float:
        return left + right

    def finalize_mean(total: float, samples: int) -> float:
        return total / samples if samples else float("nan")

    result = monte_carlo(
        10_000,
        sample_uniform,
        identity,
        init_fn=init,
        reduce_fn=reduce_sum,
        combine_fn=combine_sum,
        finalize_fn=finalize_mean,
        seed=5,
    )
    assert 0.48 < result < 0.52
