# mpipy (MVP)

This is a minimal, from-scratch MPI-style runtime in pure Python. It uses SSH for process launch and raw TCP sockets for message passing. The goal is to provide MPI-like terminology and a simple, human-readable configuration while staying small enough to extend quickly.

## Assumptions
- All nodes can SSH to each other or at least to workers from the master.
- The same project path is visible on every node (shared filesystem or identical deployment path).
- Python is installed on every node and available via the same executable name.
- The master node is reachable from workers on the TCP port chosen at runtime.
- All nodes are of identical specs (core/thread count, RAM size, etc)
## Quick Usage (In Script)
```python
from mpipy import configure_infra
from mpipy.prime import is_prime

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

print(is_prime(999983))
```
## Current Features
- The ability to calcuate primality of an int via brute force up to sqrt(n)
- Job lock: only one job can run at a time. 


## Current Limitations
- Centralized routing: all messages flow through rank 0.
- No non-blocking ops, no derived datatypes, no fault tolerance.
- Security is minimal (raw TCP); rely on cluster network isolation.

## Cancellation
Cancellation is cooperative. Call `mpipy.cancel_job()` from the master process, and ensure long-running code periodically checks `mpipy.cancel_requested()` (or `mpipy.raise_if_cancelled()`) to exit early.

## Next Steps
- Add `send/recv` tags and `source` matching in `Comm.recv`.
- Add non-blocking operations and collectives.
- Add better cluster failure detection and timeouts.
- Add proper handling of floats
- Add simpler job cancellation, preferably via terminal/cli.

