"""Example of configuring infra and running distributed matmul."""

import numpy as np

from mpipy import configure_infra
from mpipy.matmul import mat_mul


def main():
    configure_infra(
        master_node="192.168.0.1",
        per_node_cores=3,
        per_node_threads=6,
        num_worker_nodes=2,
        hosts=["192.168.0.2", "192.168.0.3"],
        time_job=True,
        progress_to_terminal=True,
        ssh_user="youruser",
        python_executable="python",
        working_dir="/shared/yourproject",
    )

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.arange(20, dtype=np.float64).reshape(4, 5)
    result = mat_mul(a, b)
    print(result)


if __name__ == "__main__":
    main()
