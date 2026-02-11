# mpipy
Some time ago, I built a beowulf cluster with four worker nodes and a master node out of sff desktops. It was a lot of fun. I tried both python and cpp MPI libraries, but I felt like they were too complicated for someone who is not already at least of intermediate MPI skill. 

This is a minimal, MPI runtime in pure Python. It uses SSH for process launch and raw TCP sockets for message passing. The goal is to utilitize the MPI spec with simple, human-readable configuration. That is to say, backend complexity in exchange for ux simplicity.

## Assumptions
- All nodes can SSH to each other or at least to workers from the master.
- The same project path is visible on every node (shared filesystem or identical deployment path).
- Python is installed on every node and available via the same executable name.
- The master node is reachable from workers on the TCP port chosen at runtime.
- All nodes are of identical specs (core/thread count, RAM size, etc)

## Usage
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
a = # m x n matrix as np.array
b = # n x e matrix as np.array 

print(is_prime(999983))
print(mat_mul(a,b))
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
- Primality testing needs to end if a rank finds a divisor while the other ranks are still searching.
- Add `send/recv` tags and `source` matching in `Comm.recv`.
- Add non-blocking operations and collectives.
- Add better cluster failure detection and timeouts.
- Add proper handling of floats
- Add simpler job cancellation, preferably via terminal/cli.
- Add video transcoding, chunk-based compression, AES encryption, neural net training, numerical differentiation, high-dimensional numerical integration, PDE solving support. 

## Ongoing
- Ensure cross-node communication is at a bare minimum. 

## Contributing
- Please feel free to contribute if you find this library and its usecase interesting, compelling, or worthwhile. 

## Disclaimer, of sorts. 
-This is my first foray into building a python library. As such, GPT Codex gave me outlines for overall structure, library requirements (presence of a .toml, etc), verification of math, and boilerplate things like unit tests. 