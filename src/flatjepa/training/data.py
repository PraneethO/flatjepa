"""Torch adapters over the F4 dataset.

:class:`~flatjepa.data.dataset.WindowedDataset` is deliberately torch-free so the data layer can be
exercised without a torch install. This module is the thin bridge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from flatjepa.data.dataset import WindowedDataset

_TENSOR_KEYS = ("state_hist", "action_hist", "action_future", "state_future", "targets")


class TorchWindowDataset(Dataset):
    """Wraps a built split. Memory-mapped, so worker processes share page cache rather than RAM."""

    def __init__(self, root: str | Path, split: str):
        self.inner = WindowedDataset(root, split)
        self.root = Path(root)
        self.split = split

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        item = self.inner[i]
        return {k: torch.from_numpy(np.ascontiguousarray(item[k])).float() for k in _TENSOR_KEYS}

    # --- passthroughs the training/probe code needs ---

    @property
    def metadata(self) -> dict:
        return self.inner.metadata

    @property
    def obs_dim(self) -> int:
        return int(self.inner.arrays["state_hist"].shape[-1])

    @property
    def history(self) -> int:
        return int(self.inner.arrays["state_hist"].shape[1])

    @property
    def horizon(self) -> int:
        return int(self.inner.arrays["action_future"].shape[1])

    def target(self, name: str) -> np.ndarray:
        return self.inner.target(name)

    def flat_inputs(self) -> np.ndarray:
        return self.inner.flat_inputs()


class GPUResidentSplit:
    """The whole split held as device tensors, iterated by index permutation.

    The built corpus is ~57 MB total, so every split fits in GPU memory many times over. Under
    those conditions a ``DataLoader`` is pure overhead: it pays per-item Python indexing, collation,
    and a host-to-device copy every batch, and it competes for CPU with the planner workers that run
    concurrently during data generation. Holding the tensors on device and slicing a shuffled index
    removes all of that.

    Falls back to CPU tensors transparently when no GPU is present, which keeps the tests honest.
    """

    def __init__(self, root: str | Path, split: str, device: torch.device | str = "cpu"):
        inner = WindowedDataset(root, split, mmap=False)
        self.device = torch.device(device)
        self.split = split
        self.metadata = inner.metadata
        self._inner = inner
        self.tensors: dict[str, torch.Tensor] = {
            k: torch.from_numpy(np.ascontiguousarray(inner.arrays[k])).float().to(self.device)
            for k in _TENSOR_KEYS
        }

    def __len__(self) -> int:
        return int(self.tensors["state_hist"].shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.tensors["state_hist"].shape[-1])

    @property
    def history(self) -> int:
        return int(self.tensors["state_hist"].shape[1])

    @property
    def horizon(self) -> int:
        return int(self.tensors["action_future"].shape[1])

    def nbytes(self) -> int:
        return sum(t.element_size() * t.nelement() for t in self.tensors.values())

    def iter_batches(
        self, batch_size: int, shuffle: bool, generator: torch.Generator | None = None
    ):
        n = len(self)
        if shuffle:
            order = torch.randperm(n, device=self.device, generator=generator)
        else:
            order = torch.arange(n, device=self.device)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            yield {k: v[idx] for k, v in self.tensors.items()}

    def target(self, name: str) -> np.ndarray:
        return self._inner.target(name)

    def flat_inputs(self) -> np.ndarray:
        return self._inner.flat_inputs()


def make_loader(
    dataset: TorchWindowDataset,
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int | None = None,
    drop_last: bool = False,
) -> DataLoader:
    """DataLoader with a seeded generator, so shuffling is reproducible from the run seed."""
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
