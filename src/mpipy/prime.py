"""Parallel primality test by checking divisibility up to sqrt(n)."""

from __future__ import annotations

import math
import os
from typing import Optional

from .config import get_config
from .runtime import COMM_WORLD, LocalComm, cancel_requested, run


def _is_prime_impl(n: int, comm) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2

    limit = int(math.isqrt(n))
    if limit < 2:
        return True

    root = 0
    size = comm.size
    span = max(0, limit - 1)
    chunk = (span + size - 1) // size
    start = 2 + comm.rank * chunk
    end = min(limit, start + chunk - 1)

    local_is_composite = False
    if start <= end:
        if start % 2 == 0:
            start += 1
        for i, d in enumerate(range(start, end + 1, 2), start=start):
            if i % 1024 == 0 and cancel_requested():
                return False
            if n % d == 0:
                local_is_composite = True
                break

    results = comm.gather(local_is_composite, root=root)
    if comm.rank == root:
        return not any(results)
    return False


def _is_prime_entry(n: int) -> bool:
    comm = COMM_WORLD or LocalComm()
    return _is_prime_impl(n, comm)


def is_prime(n: int, comm: Optional[object] = None) -> bool:
    if comm is None:
        comm = COMM_WORLD
    if comm is None:
        cfg = get_config()
        if cfg is not None and os.environ.get("MPI_RANK") is None:
            return run(_is_prime_entry, n)
        comm = LocalComm()
    return _is_prime_impl(n, comm)


def segmented_sieve(n: int, comm: Optional[object] = None) -> bool:
    return is_prime(n, comm=comm)
