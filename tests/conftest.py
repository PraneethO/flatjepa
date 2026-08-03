"""Shared pytest configuration.

Torch defaults to one intra-op thread per core. Trajectory generation runs 14 planner workers
pinned across the same cores, so during a generation run the two oversubscribe badly and the suite
slows by roughly two orders of magnitude — 11 s idle became 12 min 45 s under load average 27, which
looks indistinguishable from a hang.

The tests are small enough that single-threaded torch costs nothing when the machine is idle, so we
pin it unconditionally rather than trying to detect contention.
"""

import os

# Must be set before torch initializes its thread pools.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: optimisation-heavy test; deselect with -m 'not slow'"
    )
