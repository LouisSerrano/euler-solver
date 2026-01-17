#!/usr/bin/env python3
"""
Utility script to inspect generated HDF5 data files
"""

import h5py
import numpy as np
import argparse
from pathlib import Path


def inspect_hdf5(filename):
    """Inspect and print information about HDF5 file"""

    if not Path(filename).exists():
        print(f"Error: File '{filename}' not found!")
        return

    print("="*70)
    print(f"Inspecting: {filename}")
    print("="*70)

    with h5py.File(filename, 'r') as f:
        # Print file size
        file_size = Path(filename).stat().st_size / (1024**3)  # GB
        print(f"\nFile size: {file_size:.2f} GB")

        # Print datasets
        print("\n" + "-"*70)
        print("Datasets:")
        print("-"*70)

        if 'euler' in f:
            euler_data = f['euler']
            print(f"\nEuler trajectories:")
            print(f"  Shape: {euler_data.shape}")
            print(f"  Dtype: {euler_data.dtype}")
            print(f"  Size: {euler_data.nbytes / (1024**2):.2f} MB")
            print(f"  Min/Max: [{euler_data[0].min():.4f}, {euler_data[0].max():.4f}]")

        if 'diffusion' in f:
            diffusion_data = f['diffusion']
            print(f"\nDiffusion trajectories:")
            print(f"  Shape: {diffusion_data.shape}")
            print(f"  Dtype: {diffusion_data.dtype}")
            print(f"  Size: {diffusion_data.nbytes / (1024**2):.2f} MB")
            print(f"  Min/Max: [{diffusion_data[0].min():.4f}, {diffusion_data[0].max():.4f}]")

        if 'navier_stokes' in f:
            ns_data = f['navier_stokes']
            print(f"\nNavier-Stokes trajectories:")
            print(f"  Shape: {ns_data.shape}")
            print(f"  Dtype: {ns_data.dtype}")
            print(f"  Size: {ns_data.nbytes / (1024**2):.2f} MB")
            print(f"  Min/Max: [{ns_data[0].min():.4f}, {ns_data[0].max():.4f}]")

        # Print parameters
        print("\n" + "-"*70)
        print("Parameters:")
        print("-"*70)

        if 'parameters' in f:
            params = f['parameters']
            print(f"\nGrid resolution: {params.attrs['N']} × {params.attrs['N']}")
            print(f"Domain size: {params.attrs['L']:.4f}")
            print(f"Simulation time: {params.attrs['T']:.4f}")
            print(f"Time step: {params.attrs['dt']:.4f}")
            print(f"Snapshots per trajectory: {params.attrs['n_snapshots']}")

            if 'viscosities' in params:
                viscosities = params['viscosities'][:]
                print(f"\nViscosities ({len(viscosities)} values):")
                print(f"  Range: [{viscosities.min():.2e}, {viscosities.max():.2e}]")
                print(f"  Values: {[f'{v:.2e}' for v in viscosities]}")

        # Print metadata
        print("\n" + "-"*70)
        print("Metadata:")
        print("-"*70)

        if 'metadata' in f:
            meta = f['metadata']
            print(f"\nGeneration settings:")
            if 'n_euler' in meta.attrs:
                print(f"  Euler trajectories: {meta.attrs['n_euler']}")
            if 'n_diffusion_per_euler' in meta.attrs:
                print(f"  Diffusion per Euler: {meta.attrs['n_diffusion_per_euler']}")
            if 'n_ns_per_viscosity' in meta.attrs:
                print(f"  NS per viscosity: {meta.attrs['n_ns_per_viscosity']}")
            if 'generation_time' in meta.attrs:
                print(f"  Generation time: {meta.attrs['generation_time']:.2f} seconds")
            if 'device' in meta.attrs:
                print(f"  Device used: {meta.attrs['device']}")

        # Calculate total trajectories
        print("\n" + "-"*70)
        print("Summary:")
        print("-"*70)

        total = 0
        if 'euler' in f:
            total += f['euler'].shape[0]
        if 'diffusion' in f:
            total += f['diffusion'].shape[0]
        if 'navier_stokes' in f:
            total += f['navier_stokes'].shape[0]

        print(f"\nTotal trajectories: {total:,}")

        if 'parameters' in f:
            n_snapshots = f['parameters'].attrs['n_snapshots']
            N = f['parameters'].attrs['N']
            total_snapshots = total * n_snapshots
            total_points = total_snapshots * N * N

            print(f"Total snapshots: {total_snapshots:,}")
            print(f"Total data points: {total_points:,}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Inspect HDF5 trajectory data file'
    )
    parser.add_argument('filename', type=str, nargs='?',
                       default='data/trajectories.h5',
                       help='HDF5 file to inspect (default: data/trajectories.h5)')

    args = parser.parse_args()
    inspect_hdf5(args.filename)


if __name__ == '__main__':
    main()
