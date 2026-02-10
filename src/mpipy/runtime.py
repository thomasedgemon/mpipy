"""MPI-style runtime API and collective operations."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Optional

from .config import ConfigError, InfraConfig, clear_config, get_config
from .launcher import launch_workers
from .transport import CANCEL_TAG, MasterRouter, WorkerTransport, connect_to_master


TAG_USER = 0
TAG_BCAST = 1
TAG_SCATTER = 2
TAG_GATHER = 3
TAG_BARRIER = 4

COMM_WORLD = None

_JOB_LOCK = threading.Lock()
_JOB_ACTIVE = False
_CANCEL_EVENT = threading.Event()


class JobStateError(RuntimeError):
    pass


class JobCancelled(RuntimeError):
    pass


class Comm:
    def __init__(self, rank: int, size: int, transport):
        self.rank = rank
        self.size = size
        self._transport = transport

    def send(self, obj, dest: int, tag: int = TAG_USER):
        self._transport.send(dest=dest, tag=tag, obj=obj)

    def recv(self, source: Optional[int] = None, tag: Optional[int] = None, timeout: Optional[float] = None):
        msg = self._transport.recv(tag=tag, timeout=timeout)
        if source is not None and msg.src != source:
            return self.recv(source=source, tag=tag, timeout=timeout)
        return msg.payload

    def bcast(self, value, root: int = 0):
        if self.rank == root:
            for r in range(self.size):
                if r != root:
                    self.send(value, dest=r, tag=TAG_BCAST)
            return value
        return self.recv(source=root, tag=TAG_BCAST)

    def scatter(self, values, root: int = 0):
        if self.rank == root:
            if len(values) != self.size:
                raise ValueError("scatter values must match size")
            for r in range(self.size):
                if r != root:
                    self.send(values[r], dest=r, tag=TAG_SCATTER)
            return values[root]
        return self.recv(source=root, tag=TAG_SCATTER)

    def gather(self, value, root: int = 0):
        if self.rank == root:
            results = [None] * self.size
            results[root] = value
            for r in range(1, self.size):
                results[r] = self.recv(source=r, tag=TAG_GATHER)
            return results
        self.send(value, dest=root, tag=TAG_GATHER)
        return None

    def barrier(self):
        if self.rank == 0:
            for r in range(1, self.size):
                self.recv(source=r, tag=TAG_BARRIER)
            for r in range(1, self.size):
                self.send(True, dest=r, tag=TAG_BARRIER)
            return
        self.send(True, dest=0, tag=TAG_BARRIER)
        self.recv(source=0, tag=TAG_BARRIER)


class LocalComm:
    rank = 0
    size = 1

    def send(self, obj, dest: int, tag: int = TAG_USER):
        raise RuntimeError("send not available in LocalComm")

    def recv(self, source: Optional[int] = None, tag: Optional[int] = None, timeout: Optional[float] = None):
        raise RuntimeError("recv not available in LocalComm")

    def bcast(self, value, root: int = 0):
        return value

    def scatter(self, values, root: int = 0):
        return values[0]

    def gather(self, value, root: int = 0):
        return [value]

    def barrier(self):
        return


def _env_rank() -> Optional[int]:
    val = os.environ.get("MPI_RANK")
    if val is None:
        return None
    return int(val)


def init() -> Comm:
    rank = _env_rank()
    if rank is None:
        raise ConfigError("MPI_RANK not set; use run() or mpipy-run")
    size = int(os.environ["MPI_WORLD_SIZE"])
    host = os.environ["MPI_MASTER_HOST"]
    port = int(os.environ["MPI_MASTER_PORT"])
    transport = connect_to_master(host, port, rank, _CANCEL_EVENT)
    comm = Comm(rank=rank, size=size, transport=transport)
    global COMM_WORLD
    COMM_WORLD = comm
    return comm


def init_master(cfg: InfraConfig, module: str, function: str, args, kwargs) -> Comm:
    expected_workers = cfg.num_worker_nodes * cfg.per_node_cores
    router = MasterRouter(cfg.master_node, 0, expected_workers=expected_workers)
    world_size = launch_workers(cfg, cfg.master_node, router.actual_port, module, function, args, kwargs)
    router.accept_all(cfg.connect_timeout_s)
    comm = Comm(rank=0, size=world_size, transport=router)
    global COMM_WORLD
    COMM_WORLD = comm
    return comm


def cancel_job() -> None:
    if COMM_WORLD is None or not _JOB_ACTIVE:
        raise JobStateError("No active job to cancel")
    _CANCEL_EVENT.set()
    if COMM_WORLD.rank == 0:
        transport = COMM_WORLD._transport
        for r in range(1, COMM_WORLD.size):
            transport.send_control(dest=r, tag=CANCEL_TAG, obj=None)


def cancel_requested() -> bool:
    return _CANCEL_EVENT.is_set()


def raise_if_cancelled() -> None:
    if _CANCEL_EVENT.is_set():
        raise JobCancelled("Job was cancelled")


def run(fn: Callable[..., Any], *args, **kwargs):
    global _JOB_ACTIVE, COMM_WORLD
    cfg = get_config()
    if cfg is None:
        raise ConfigError("configure_infra must be called before run")

    if _env_rank() is not None:
        if COMM_WORLD is None:
            init()
        return fn(*args, **kwargs)

    with _JOB_LOCK:
        if _JOB_ACTIVE:
            raise JobStateError("A job is already running; wait for it to finish before starting a new one")
        _JOB_ACTIVE = True
        _CANCEL_EVENT.clear()

    start = time.time() if cfg.time_job else None
    module = fn.__module__
    function = fn.__name__
    comm = init_master(cfg, module, function, args, kwargs)
    try:
        result = fn(*args, **kwargs)
    finally:
        comm.barrier()
        COMM_WORLD = None
        clear_config()
        _CANCEL_EVENT.clear()
        with _JOB_LOCK:
            _JOB_ACTIVE = False
    if start is not None:
        elapsed = time.time() - start
        return {"result": result, "elapsed_s": elapsed}
    return result
