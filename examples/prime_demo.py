"""Example of configuring infra and running the segmented sieve."""

from mpipy import configure_infra
from mpipy.prime import is_prime


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

    result = is_prime(99999999999983)
    print(result)


if __name__ == "__main__":
    main()
