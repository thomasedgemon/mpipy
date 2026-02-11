"""Parallel matrix multiplication using a 2D block decomposition."""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np

from .config import get_config
from .runtime import COMM_WORLD, LocalComm, cancel_requested, run


_TAG_A_BASE = 1000
_TAG_B_BASE = 2000
_TAG_A_STEP_BASE = 3000
_TAG_B_STEP_BASE = 4000

_MATMUL_INPUTS: Optional[Tuple[np.ndarray, np.ndarray]] = None


def _as_2d_array(value, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    return arr


def _partition_ranges(n: int, parts: int) -> list[Tuple[int, int]]:
    base = n // parts
    remainder = n % parts
    ranges: list[Tuple[int, int]] = []
    start = 0
    for i in range(parts):
        size = base + (1 if i < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def _grid_dims(size: int) -> Tuple[int, int]:
    root = int(math.sqrt(size))
    for pr in range(root, 0, -1):
        if size % pr == 0:
            return pr, size // pr
    return 1, size


def _matmul_impl(a: np.ndarray, b: np.ndarray, comm) -> Optional[np.ndarray]:
    a = _as_2d_array(a, "a")
    b = _as_2d_array(b, "b")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible matrix dimensions")
    if comm.size == 1:
        return a @ b
    return _matmul_distributed(comm, a=a, b=b)


def _matmul_distributed(comm, a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if comm.size == 1:
        if a is None or b is None:
            return None
        return a @ b

    pr, pc = _grid_dims(comm.size)

    if comm.rank == 0:
        if a is None or b is None:
            raise ValueError("root rank requires input matrices")
        a = _as_2d_array(a, "a")
        b = _as_2d_array(b, "b")
        if a.shape[1] != b.shape[0]:
            raise ValueError("incompatible matrix dimensions")
        m, k = a.shape
        _k2, n = b.shape
        dtype = np.result_type(a, b)
    else:
        a = None
        b = None
        m = k = n = 0
        dtype = None

    if comm.rank == 0:
        meta = (m, k, n, dtype)
    else:
        meta = None
    m, k, n, dtype = comm.bcast(meta, root=0)

    row_ranges = _partition_ranges(m, pr)
    k_ranges = _partition_ranges(k, pc)
    col_ranges = _partition_ranges(n, pc)

    r = comm.rank // pc
    c = comm.rank % pc

    row_start, row_end = row_ranges[r]
    col_start, col_end = col_ranges[c]

    local_rows = row_end - row_start
    local_cols = col_end - col_start
    local_c = np.zeros((local_rows, local_cols), dtype=dtype)

    local_b_blocks: dict[int, np.ndarray] = {}
    if comm.rank == 0:
        for rr in range(pr):
            rs, re = row_ranges[rr]
            for cc in range(pc):
                ks, ke = k_ranges[cc]
                a_block = a[rs:re, ks:ke]
                dest = rr * pc + cc
                if dest == 0:
                    local_a = a_block
                else:
                    comm.send(a_block, dest=dest, tag=_TAG_A_BASE + dest)
        for q in range(pc):
            ks, ke = k_ranges[q]
            owner_row = q % pr
            for cc in range(pc):
                cs, ce = col_ranges[cc]
                b_block = b[ks:ke, cs:ce]
                dest = owner_row * pc + cc
                if dest == 0:
                    local_b_blocks[q] = b_block
                else:
                    comm.send(b_block, dest=dest, tag=_TAG_B_BASE + q)
    else:
        local_a = comm.recv(source=0, tag=_TAG_A_BASE + comm.rank)
        for q in range(pc):
            owner_row = q % pr
            if r == owner_row:
                local_b_blocks[q] = comm.recv(source=0, tag=_TAG_B_BASE + q)

    for q in range(pc):
        if cancel_requested():
            return None

        if c == q:
            a_panel = local_a
            for dest_c in range(pc):
                if dest_c == c:
                    continue
                dest_rank = r * pc + dest_c
                comm.send(a_panel, dest=dest_rank, tag=_TAG_A_STEP_BASE + q)
        else:
            owner_rank = r * pc + q
            a_panel = comm.recv(source=owner_rank, tag=_TAG_A_STEP_BASE + q)

        owner_row = q % pr
        if r == owner_row:
            b_panel = local_b_blocks[q]
            for dest_r in range(pr):
                if dest_r == r:
                    continue
                dest_rank = dest_r * pc + c
                comm.send(b_panel, dest=dest_rank, tag=_TAG_B_STEP_BASE + q)
        else:
            owner_rank = owner_row * pc + c
            b_panel = comm.recv(source=owner_rank, tag=_TAG_B_STEP_BASE + q)

        if a_panel.size and b_panel.size:
            local_c += a_panel @ b_panel

    gathered = comm.gather(local_c, root=0)
    if comm.rank != 0:
        return None

    result = np.zeros((m, n), dtype=dtype)
    for rank, block in enumerate(gathered):
        if block is None:
            continue
        rr = rank // pc
        cc = rank % pc
        rs, re = row_ranges[rr]
        cs, ce = col_ranges[cc]
        if rs == re or cs == ce:
            continue
        result[rs:re, cs:ce] = block
    return result


def _matmul_entry(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    comm = COMM_WORLD or LocalComm()
    return _matmul_impl(a, b, comm)


def _matmul_distributed_entry() -> Optional[np.ndarray]:
    comm = COMM_WORLD or LocalComm()
    if comm.rank == 0:
        if _MATMUL_INPUTS is None:
            raise ValueError("matmul inputs missing on root")
        a, b = _MATMUL_INPUTS
    else:
        a = b = None
    return _matmul_distributed(comm, a=a, b=b)


def mat_mul(a: np.ndarray, b: np.ndarray, comm: Optional[object] = None) -> Optional[np.ndarray]:
    if comm is None:
        comm = COMM_WORLD
    if comm is None:
        cfg = get_config()
        if cfg is not None and os.environ.get("MPI_RANK") is None:
            global _MATMUL_INPUTS
            _MATMUL_INPUTS = (a, b)
            try:
                return run(_matmul_distributed_entry)
            finally:
                _MATMUL_INPUTS = None
        comm = LocalComm()
    return _matmul_impl(a, b, comm)
