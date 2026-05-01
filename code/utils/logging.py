"""Tiny logging helper that prefixes messages with the script name."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path


def _prefix() -> str:
    return Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "pipeline"


def info(msg: str) -> None:
    sys.stderr.write(f"[{_prefix()}] {msg}\n")
    sys.stderr.flush()


@contextmanager
def stage(name: str):
    info(f"START {name}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        info(f"DONE  {name} in {time.perf_counter() - t0:.1f}s")
