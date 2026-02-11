# Documentation

This document describes the functions in `mpipy`, based on:
- `src/mpipy/matmul.py`
- `src/mpipy/prime.py`
- `src/mpipy/monte_carlo.py`

All three modules follow the same execution pattern:
- If `comm` is provided, they use it directly.
- If `comm` is not provided and `configure_infra(...)` has been called, they launch a distributed job via `mpipy.run(...)`.
- Otherwise they fall back to `LocalComm()` and run locally.

Only rank 0 returns the final result in distributed mode. Other ranks return `None`.

---

**Matrix Multiplication**

**Function**
`mat_mul(a: np.ndarray, b: np.ndarray, comm: Optional[object] = None) -> Optional[np.ndarray]`

**Behavior**
- Validates `a` and `b` as 2D arrays and checks compatibility (`a.shape[1] == b.shape[0]`).
- Uses a 2D block decomposition across ranks.
- Returns the full product matrix on rank 0, `None` on other ranks.
- When running locally, returns `a @ b`.

**Notes**
- Uses `numpy` and returns `np.ndarray`.
- Cancels cooperatively by checking `cancel_requested()` in the main loop.

**Example**
```python
from mpipy.matmul import mat_mul

c = mat_mul(a, b)
```

---

**Primality Test**

**Function**
`is_prime(n: int, comm: Optional[object] = None) -> bool`

**Behavior**
- Checks divisibility up to `sqrt(n)` using odd divisors.
- Splits the divisor range evenly across ranks.
- Each rank computes whether its segment finds a factor.
- Rank 0 gathers results and returns `True` only if no rank found a divisor.
- Other ranks return `False` in distributed mode.

**Notes**
- Returns a boolean in local mode.
- Cancellation is checked every 1024 iterations.

**Example**
```python
from mpipy.prime import is_prime

flag = is_prime(999983)
```

---

**Monte Carlo**

**Function**
`monte_carlo(num_samples, sample_fn, eval_fn, *, init_fn=None, reduce_fn=None, combine_fn=None, finalize_fn=None, seed=None, cancel_check_every=1024, comm=None)`

**Behavior**
- Partitions `num_samples` evenly across ranks.
- Each rank runs a local loop:
  - `sample_fn(rng)` produces a sample from a `random.Random` instance.
  - `eval_fn(sample)` maps the sample to a numeric value (default reducer).
- Local accumulators are gathered to rank 0, combined, and finalized.

**Default Reducer**
If `reduce_fn` is not provided, the default reducer computes:
- mean
- variance
- standard error

Return type: `MonteCarloResult(mean, variance, stderr, samples)`

**Custom Reducer**
If `reduce_fn` is provided, you must also provide:
- `init_fn() -> acc`
- `reduce_fn(acc, value) -> acc`
- `combine_fn(left, right) -> acc`

`finalize_fn` is optional. If omitted, the combined accumulator is returned on rank 0.

**Example (default reducer)**
```python
import random
from mpipy.monte_carlo import monte_carlo

def sample(rng: random.Random):
    return rng.random()

def identity(x: float) -> float:
    return x

result = monte_carlo(100_000, sample, identity, seed=1234)
```

**Example (custom reducer)**
```python
import random
from mpipy.monte_carlo import monte_carlo

def sample(rng: random.Random):
    return rng.random()

def identity(x: float) -> float:
    return x

def init():
    return 0.0

def reduce_sum(acc: float, value: float) -> float:
    return acc + value

def combine_sum(left: float, right: float) -> float:
    return left + right

def finalize_mean(total: float, samples: int) -> float:
    return total / samples if samples else float("nan")

mean = monte_carlo(
    10_000,
    sample,
    identity,
    init_fn=init,
    reduce_fn=reduce_sum,
    combine_fn=combine_sum,
    finalize_fn=finalize_mean,
    seed=5,
)
```

---

**Return Semantics**
- Local mode: returns the computed value directly.
- Distributed mode: rank 0 returns the result, all other ranks return `None` (or `False` for `is_prime`).

If you want all ranks to receive the result, call `comm.bcast(...)` yourself after rank 0 computes it.
