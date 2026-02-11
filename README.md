# mpipy

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
from mpipy.matmul import mat_mul
from mpipy.monte_carlo import monte_carlo

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
print(mat_mul(a,b)) #where a and b are type np.array
print(monte_carlo(10000, sample_fn, eval_fn))
```
## Current Features
- The ability to naively calcuate primality of an integer via brute force up to sqrt(n), skipping even numbers. The library automatically splits up the range (1-sqrt(n)) evenly across all specified worker nodes. 
- The ability to distribute matrix multiplication via 2d block decomposition without square grid
restrictions in order to distribute the work as evenly as possible. 
- The ability to distribute generic Monte Carlo support with customizable reducers for estimators.
- Job lock: only one job can run at a time. 


## Current Limitations
- Centralized routing: all messages flow through rank 0 (the master node).
- No non-blocking ops, no derived datatypes, no fault tolerance.
- Security is minimal (raw TCP); rely on cluster network isolation.

## Cancellation
Cancellation is cooperative. Call `mpipy.cancel_job()` from the master process, and ensure long-running code periodically checks `mpipy.cancel_requested()` (or `mpipy.raise_if_cancelled()`) to exit early.

## Additional Requirements
Numpy


## Notes
It is suggested to use at most (total cores - 1) on a node, to leave room for OS processes, etc. 

## Next Steps
- Add `send/recv` tags and `source` matching in `Comm.recv`.
- Add non-blocking operations and collectives.
- Add better cluster failure detection and timeouts.
- Add proper handling of floats
- Add simpler job cancellation, preferably via terminal/cli.
- Add video transcoding, chunk-based compression, AES encryption, neural net training, numerical differentiation, high-dimensional numerical integration, PDE solving support. 

## Ongoing
- Ensure cross-node communication is at a bare minimum. 