"""Example of configuring infra and running distributed monte carlo."""
import math
import random

from mpipy import configure_infra
from mpipy.monte_carlo import MonteCarloResult, monte_carlo


def sample_unit_square(rng: random.Random) -> tuple[float, float]:
    return rng.random(), rng.random()


def inside_unit_circle(point: tuple[float, float]) -> float:
    x, y = point
    return 1.0 if x * x + y * y <= 1.0 else 0.0


if __name__ == "__main__":
    configure_infra(
        master_node="headnode",
        per_node_cores=8,
        per_node_threads=16,
        num_worker_nodes=2,
        hosts=["worker1ip", "worker2ip"],
        time_job=True,
        progress_to_terminal=True,
        ssh_user="youruser",
        python_executable="python",
        working_dir="/shared/yourproject",
    )

    result = monte_carlo(200_000, sample_unit_square, inside_unit_circle, seed=1234)
    if isinstance(result, MonteCarloResult):
        pi_estimate = 4.0 * result.mean
        print(f"pi ≈ {pi_estimate:.6f} (stderr {4.0 * result.stderr:.6f})")
