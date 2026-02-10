"""SSH-based launcher for starting worker processes on remote nodes."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Optional

from .config import InfraConfig
from .transport import encode_args


class LaunchError(RuntimeError):
    pass


def _ssh_prefix(cfg: InfraConfig, host: str) -> list[str]:
    user_host = f"{cfg.ssh_user}@{host}" if cfg.ssh_user else host
    cmd = ["ssh"]
    if cfg.ssh_port:
        cmd.extend(["-p", str(cfg.ssh_port)])
    if cfg.ssh_identity_file:
        cmd.extend(["-i", cfg.ssh_identity_file])
    cmd.append(user_host)
    return cmd


def launch_workers(cfg: InfraConfig, master_host: str, master_port: int, module: str, function: str, args, kwargs):
    if not cfg.hosts:
        raise LaunchError("hosts list is required for SSH launch")

    encoded_args = encode_args(args, kwargs)
    ranks_per_node = cfg.per_node_cores
    world_size = cfg.num_worker_nodes * ranks_per_node + 1

    rank = 1
    for host in cfg.hosts:
        for local_rank in range(ranks_per_node):
            env = {
                "MPI_MASTER_HOST": master_host,
                "MPI_MASTER_PORT": str(master_port),
                "MPI_WORLD_SIZE": str(world_size),
                "MPI_RANK": str(rank),
                "MPI_RUN_MODULE": module,
                "MPI_RUN_FUNCTION": function,
                "MPI_RUN_ARGS": encoded_args,
            }
            export = " ".join(f"{k}='{v}'" for k, v in env.items())
            python = cfg.python_executable or "python"
            workdir = f"cd '{cfg.working_dir}' && " if cfg.working_dir else ""
            remote_cmd = f"{workdir}{export} {python} -m mpipy.worker"
            cmd = _ssh_prefix(cfg, host) + [remote_cmd]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if cfg.progress_to_terminal:
                print(f"[mpipy] launched rank {rank} on {host} (local {local_rank})")
            rank += 1
            time.sleep(0.05)
    return world_size
