"""
Dataset classes for Euler/Diffusion/Navier-Stokes trajectory data.

Loads directly from split GPU files (no merge required).

Training/Validation: Euler + Diffusion trajectories
Testing: Navier-Stokes trajectories
"""
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from typing import List, Union, Optional


class MultiFileHDF5Dataset:
    """
    Base class for loading trajectories from multiple HDF5 files.
    Keeps file handles open for fast access.
    """

    def __init__(self, file_paths: List[str], keep_file_open: bool = True):
        self.file_paths = [str(p) for p in file_paths]
        self.keep_file_open = keep_file_open
        self.file_handles = {}

        if keep_file_open:
            for path in self.file_paths:
                self.file_handles[path] = h5py.File(path, 'r')

    def _get_file(self, path: str):
        """Get file handle, opening if necessary."""
        if self.keep_file_open:
            return self.file_handles[path], False
        return h5py.File(path, 'r'), True

    def close(self):
        """Close all file handles."""
        for f in self.file_handles.values():
            f.close()
        self.file_handles = {}

    def __del__(self):
        self.close()


class EulerDiffusionDataset(MultiFileHDF5Dataset, Dataset):
    """
    Dataset for training/validation on Euler and Diffusion trajectories.
    Loads directly from split GPU files.

    Labels:
        - 0: Euler (nu=0)
        - 1, 2, ..., m_visc: Diffusion with viscosity index

    Args:
        file_dir: Directory containing trajectories_gpu*.h5 files
        num_gpus: Number of GPU files
        split: 'train' or 'val'
        val_fraction: Fraction of data for validation
        seed: Random seed for train/val split
    """

    def __init__(self, file_dir: str, num_gpus: int, split: str = 'train',
                 val_fraction: float = 0.1, seed: int = 42):
        # Build file paths
        file_dir = Path(file_dir)
        file_paths = [file_dir / f"trajectories_gpu{i}.h5" for i in range(num_gpus)]

        super().__init__(file_paths)

        self.split = split

        # Read metadata from first file
        f0, should_close = self._get_file(str(file_paths[0]))
        self.viscosities = f0.attrs['viscosities'][:]
        self.n_snapshots = f0.attrs['n_snapshots']
        self.save_res = f0.attrs['save_res']
        self.m_visc = len(self.viscosities)
        if should_close:
            f0.close()

        # Build index: list of (file_path, dataset_name, local_idx, label)
        all_indices = []

        for path in self.file_paths:
            f, should_close = self._get_file(str(path))

            euler_count = f.attrs['euler_count']
            diff_count = f.attrs['diff_count']
            n_diff_per_visc = diff_count // self.m_visc

            # Euler samples (label=0)
            for i in range(euler_count):
                all_indices.append((str(path), 'euler', i, 0))

            # Diffusion samples: stored as [visc0][visc1]...[visc_m] in each file
            for v_idx in range(self.m_visc):
                start = v_idx * n_diff_per_visc
                for i in range(n_diff_per_visc):
                    all_indices.append((str(path), 'diffusion', start + i, v_idx + 1))

            if should_close:
                f.close()

        # Train/val split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_indices))
        n_val = int(len(all_indices) * val_fraction)

        if split == 'val':
            selected = indices[:n_val]
        else:
            selected = indices[n_val:]

        self.indices = [all_indices[i] for i in selected]

        print(f"EulerDiffusionDataset [{split}]: {len(self.indices)} samples "
              f"({len([x for x in self.indices if x[3] == 0])} euler, "
              f"{len([x for x in self.indices if x[3] > 0])} diffusion)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_path, dataset_name, local_idx, label = self.indices[idx]

        f, should_close = self._get_file(file_path)
        try:
            trajectory = f[dataset_name][local_idx]  # (n_snapshots, H, W)
        finally:
            if should_close:
                f.close()

        return torch.from_numpy(trajectory), label

    def get_viscosity(self, label):
        """Convert label to viscosity value. Label 0 (Euler) returns 0.0."""
        if label == 0:
            return 0.0
        return self.viscosities[label - 1]

    @property
    def num_classes(self):
        return 1 + self.m_visc


class NavierStokesDataset(MultiFileHDF5Dataset, Dataset):
    """
    Dataset for testing on Navier-Stokes trajectories.
    Loads directly from split GPU files.

    Labels: 0, 1, ..., m_visc-1 corresponding to viscosity indices.

    Args:
        file_dir: Directory containing trajectories_gpu*.h5 files
        num_gpus: Number of GPU files
        N_ns_ics: Number of ICs per viscosity (from generator config)
    """

    def __init__(self, file_dir: str, num_gpus: int, N_ns_ics: int = 512):
        file_dir = Path(file_dir)
        file_paths = [file_dir / f"trajectories_gpu{i}.h5" for i in range(num_gpus)]

        super().__init__(file_paths)

        self.N_ns_ics = N_ns_ics

        # Read metadata
        f0, should_close = self._get_file(str(file_paths[0]))
        self.viscosities = f0.attrs['viscosities'][:]
        self.n_snapshots = f0.attrs['n_snapshots']
        self.save_res = f0.attrs['save_res']
        self.m_visc = len(self.viscosities)
        if should_close:
            f0.close()

        # Build index: (file_path, local_idx, visc_label)
        # NS global structure: v_idx = global_idx // N_ns_ics
        all_indices = []

        for path in self.file_paths:
            f, should_close = self._get_file(str(path))

            ns_start = f.attrs['ns_start']
            ns_count = f.attrs['ns_count']

            for local_idx in range(ns_count):
                global_idx = ns_start + local_idx
                v_idx = global_idx // N_ns_ics
                all_indices.append((str(path), local_idx, v_idx))

            if should_close:
                f.close()

        self.indices = all_indices
        print(f"NavierStokesDataset: {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_path, local_idx, visc_label = self.indices[idx]

        f, should_close = self._get_file(file_path)
        try:
            trajectory = f['navier_stokes'][local_idx]
        finally:
            if should_close:
                f.close()

        return torch.from_numpy(trajectory), visc_label

    def get_viscosity(self, label):
        return self.viscosities[label]

    @property
    def num_classes(self):
        return self.m_visc


def get_dataloaders(file_dir: str, num_gpus: int, batch_size: int = 32,
                    num_workers: int = 4, val_fraction: float = 0.1,
                    seed: int = 42, N_ns_ics: int = 512):
    """
    Get train/val/test dataloaders from split GPU files.

    Args:
        file_dir: Directory containing trajectories_gpu*.h5 files
        num_gpus: Number of GPU files
        batch_size: Batch size
        num_workers: DataLoader workers
        val_fraction: Fraction for validation
        seed: Random seed
        N_ns_ics: NS ICs per viscosity (from generator)

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = EulerDiffusionDataset(file_dir, num_gpus, split='train',
                                          val_fraction=val_fraction, seed=seed)
    val_dataset = EulerDiffusionDataset(file_dir, num_gpus, split='val',
                                        val_fraction=val_fraction, seed=seed)
    test_dataset = NavierStokesDataset(file_dir, num_gpus, N_ns_ics=N_ns_ics)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader
