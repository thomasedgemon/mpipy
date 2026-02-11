"""Infra configuration and validation for the mpipy runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional


@dataclass
class InfraConfig:
    master_node: str
    per_node_cores: int
    per_node_threads: Optional[int]
    num_worker_nodes: int
    time_job: bool = False
    progress_to_terminal: bool = False

    hosts: List[str] = field(default_factory=list)
    hostfile: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_port: int = 22
    ssh_identity_file: Optional[str] = None
    python_executable: str = "python"
    working_dir: Optional[str] = None
    connect_timeout_s: float = 10.0

_CONFIG: Optional[InfraConfig] = None

class ConfigError(ValueError):
    pass

def _read_hostfile(path: str) -> List[str]:
    hosts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            hosts.append(line)
    return hosts

#user-inputted data is validated here and put in an instance of InfraConfig
def configure_infra(
    *,
    master_node: str,
    per_node_cores: int,
    per_node_threads: Optional[int], #rm thread specs 
    num_worker_nodes: Optional[int] = None,
    time_job: bool = False,
    progress_to_terminal: bool = False,
    hosts: Optional[Iterable[str]] = None, #should mandate at least one worker address.
    hostfile: Optional[str] = None,
    ssh_user: Optional[str] = None,
    ssh_port: int = 22,
    ssh_identity_file: Optional[str] = None,
    python_executable: str = "python",
    working_dir: Optional[str] = None,
    connect_timeout_s: float = 10.0,
) -> InfraConfig:
    """Configure cluster/runtime settings.

    This is intentionally human-readable and favors explicit configuration.
    """
    if master_node is None or master_node == "":
        raise ConfigError("master_node cannot be null")
    if per_node_cores is None:
        raise ConfigError("per_node_cores cannot be null")
    if per_node_threads is not None and per_node_threads <= 0:
        raise ConfigError("per_node_threads must be positive if set")

    host_list: List[str] = []
    if hosts:
        host_list.extend(list(hosts))
    if hostfile:
        host_list.extend(_read_hostfile(hostfile))

    if num_worker_nodes is None:
        if not host_list:
            raise ConfigError("num_worker_nodes cannot be null when hosts are not provided")
        num_worker_nodes = len(host_list)

    if num_worker_nodes <= 0:
        raise ConfigError("num_worker_nodes must be positive")
    
    if num_worker_nodes is None:
        raise ConfigError("num_worker_nodes must be specified and greater than zero")

    if host_list and len(host_list) != num_worker_nodes:
        raise ConfigError("num_worker_nodes must match number of hosts")

    cfg = InfraConfig(
        master_node=master_node,
        per_node_cores=per_node_cores,
        per_node_threads=per_node_threads,
        num_worker_nodes=num_worker_nodes,
        time_job=time_job,
        progress_to_terminal=progress_to_terminal,
        hosts=host_list,
        hostfile=hostfile,
        ssh_user=ssh_user,
        ssh_port=ssh_port,
        ssh_identity_file=ssh_identity_file,
        python_executable=python_executable,
        working_dir=working_dir,
        connect_timeout_s=connect_timeout_s,
    )

    global _CONFIG
    _CONFIG = cfg
    return cfg

def get_config() -> Optional[InfraConfig]:
    return _CONFIG

def clear_config() -> None:
    global _CONFIG
    _CONFIG = None
