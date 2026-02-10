"""Worker entrypoint that bootstraps COMM_WORLD and runs a target function."""

from __future__ import annotations

import importlib
import os
import sys

from .runtime import init
from .transport import decode_args


def worker_main():
    init()
    module = os.environ.get("MPI_RUN_MODULE")
    function = os.environ.get("MPI_RUN_FUNCTION")
    if not module or not function:
        raise RuntimeError("MPI_RUN_MODULE and MPI_RUN_FUNCTION must be set")
    encoded = os.environ.get("MPI_RUN_ARGS", "")
    args, kwargs = decode_args(encoded) if encoded else ([], {})

    mod = importlib.import_module(module)
    fn = getattr(mod, function)
    fn(*args, **kwargs)


if __name__ == "__main__":
    worker_main()
