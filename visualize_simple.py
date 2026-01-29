#!/usr/bin/env python3
"""
Simple visualization script for trajectories from trajectories_gpu0.h5
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    filepath = "../ceph/data/euler_ns/trajectories_gpu0.h5"

    # Load and inspect the file
    with h5py.File(filepath, 'r') as f:
        print("Dataset structure:")
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                print(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")

        print("\nMetadata:")
        for attr in f.attrs:
            print(f"  {attr}: {f.attrs[attr]}")

        # Load one sample from each class
        datasets = {}
        for name in ['euler', 'diffusion', 'navier_stokes']:
            if name in f:
                datasets[name] = f[name][0]  # First trajectory
                print(f"\nLoaded {name}: shape={datasets[name].shape}")

    # Plot comparison: 6 time snapshots for each class
    n_classes = len(datasets)
    fig, axes = plt.subplots(n_classes, 6, figsize=(18, 3*n_classes))

    if n_classes == 1:
        axes = axes.reshape(1, -1)

    class_labels = {'euler': 'Euler', 'navier_stokes': 'Navier-Stokes', 'diffusion': 'Diffusion'}

    for i, (name, traj) in enumerate(datasets.items()):
        n_snapshots = traj.shape[0]
        time_indices = np.linspace(0, n_snapshots-1, 6, dtype=int)

        vmax = np.abs(traj).max()
        vmin = -vmax

        for j, t_idx in enumerate(time_indices):
            im = axes[i, j].imshow(traj[t_idx], cmap='RdBu_r', origin='lower',
                                   vmin=vmin, vmax=vmax)
            axes[i, j].set_title(f't={t_idx}', fontsize=10)
            axes[i, j].axis('off')

        # Add colorbar to last column
        plt.colorbar(im, ax=axes[i, -1], fraction=0.046, pad=0.04)

        # Row label
        axes[i, 0].text(-0.1, 0.5, class_labels.get(name, name),
                        transform=axes[i, 0].transAxes, fontsize=12,
                        va='center', ha='right', fontweight='bold')

    plt.suptitle('Trajectory Comparison (Sample 0)', fontsize=14)
    plt.tight_layout()

    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/simple_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: visualizations/simple_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
