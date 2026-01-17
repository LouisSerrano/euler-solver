"""
Data generation pipeline for Euler, Diffusion, and Navier-Stokes trajectories
"""

import torch
import h5py
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import time

from solver import EulerSolver2D, DiffusionSolver2D, NavierStokesSolver2D


class TrajectoryGenerator:
    """Generate and save trajectories for Euler, Diffusion, and Navier-Stokes"""

    def __init__(self,
                 N: int = 128,
                 L: float = 2 * np.pi,
                 T: float = 10.0,
                 dt: float = 0.01,
                 n_snapshots: int = 50,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 filter_order: int = 16):
        """
        Initialize the trajectory generator

        Args:
            N: Grid resolution
            L: Domain size
            T: Simulation time per trajectory
            dt: Time step
            n_snapshots: Number of snapshots per trajectory
            device: 'cuda' or 'cpu'
        """
        self.N = N
        self.L = L
        self.T = T
        self.dt = dt
        self.n_snapshots = n_snapshots
        self.device = device

        # Initialize solvers
        self.euler_solver = EulerSolver2D(N, L, device, filter_order=filter_order)
        self.diffusion_solver = DiffusionSolver2D(N, L, device) # Not used with filter usually
        self.ns_solver = NavierStokesSolver2D(N, L, device, filter_order=filter_order)

    def generate_random_initial_condition(self, omega0: float = 1.0,
                                         n_modes: int = 5,
                                         seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate a randomized initial condition based on Taylor-Green vortex

        Args:
            omega0: Base vorticity amplitude
            n_modes: Number of Fourier modes to randomize
            seed: Random seed for reproducibility

        Returns:
            Initial vorticity field
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Start with base Taylor-Green vortex
        X, Y = self.euler_solver.X, self.euler_solver.Y
        omega = omega0 * (torch.sin(X) * torch.cos(Y) + torch.cos(X) * torch.sin(Y))

        # Add random perturbations
        for i in range(1, n_modes + 1):
            for j in range(1, n_modes + 1):
                amp = torch.randn(1, device=self.device) * omega0 / (i + j)
                phase = torch.rand(1, device=self.device) * 2 * np.pi
                omega += amp * torch.sin(i * X + phase) * torch.cos(j * Y + phase)

        return omega

    def generate_euler_trajectories(self,
                                   n_trajectories: int,
                                   omega0: float = 1.0,
                                   n_modes: int = 5,
                                   random_seed_start: int = 0) -> torch.Tensor:
        """
        Generate multiple Euler equation trajectories

        Args:
            n_trajectories: Number of trajectories to generate
            omega0: Base vorticity amplitude
            n_modes: Number of Fourier modes for randomization
            random_seed_start: Starting seed for reproducibility

        Returns:
            Tensor of shape (n_trajectories, n_snapshots, N, N)
        """
        trajectories = []

        for i in tqdm(range(n_trajectories), desc="Generating Euler trajectories"):
            # Generate random initial condition
            omega0_field = self.generate_random_initial_condition(
                omega0, n_modes, seed=random_seed_start + i
            )

            # Solve Euler equations
            trajectory = self.euler_solver.solve(
                omega0_field, self.T, self.dt, nu=0.0, n_snapshots=self.n_snapshots
            )

            trajectories.append(trajectory.cpu())

        return torch.stack(trajectories)

    def generate_diffusion_from_snapshots(self,
                                         euler_trajectories: torch.Tensor,
                                         viscosities: List[float],
                                         snapshot_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Generate diffusion trajectories from Euler snapshots with different viscosities

        Args:
            euler_trajectories: Euler trajectories of shape (n_traj, n_snapshots, N, N)
            viscosities: List of viscosity values to use
            snapshot_indices: Which snapshots to use as initial conditions (default: all)

        Returns:
            Tensor of shape (n_diffusion_traj, n_snapshots, N, N)
        """
        if snapshot_indices is None:
            snapshot_indices = list(range(euler_trajectories.shape[1]))

        n_euler_traj = euler_trajectories.shape[0]
        diffusion_trajectories = []

        total_runs = n_euler_traj * len(snapshot_indices) * len(viscosities)

        with tqdm(total=total_runs, desc="Generating diffusion trajectories") as pbar:
            for traj_idx in range(n_euler_traj):
                for snap_idx in snapshot_indices:
                    for nu in viscosities:
                        # Get initial condition from Euler snapshot
                        omega0 = euler_trajectories[traj_idx, snap_idx].to(self.device)

                        # Solve diffusion equation
                        trajectory = self.diffusion_solver.solve(
                            omega0, self.T, self.dt, nu=nu, n_snapshots=self.n_snapshots
                        )

                        diffusion_trajectories.append(trajectory.cpu())
                        pbar.update(1)

        return torch.stack(diffusion_trajectories)

    def generate_ns_trajectories(self,
                                n_trajectories: int,
                                viscosities: List[float],
                                omega0: float = 1.0,
                                n_modes: int = 5,
                                random_seed_start: int = 0) -> torch.Tensor:
        """
        Generate Navier-Stokes trajectories with varying viscosities

        Args:
            n_trajectories: Number of trajectories per viscosity
            viscosities: List of viscosity values
            omega0: Base vorticity amplitude
            n_modes: Number of Fourier modes for randomization
            random_seed_start: Starting seed for reproducibility

        Returns:
            Tensor of shape (n_trajectories * len(viscosities), n_snapshots, N, N)
        """
        ns_trajectories = []
        total_runs = n_trajectories * len(viscosities)

        with tqdm(total=total_runs, desc="Generating Navier-Stokes trajectories") as pbar:
            for i in range(n_trajectories):
                # Generate random initial condition
                omega0_field = self.generate_random_initial_condition(
                    omega0, n_modes, seed=random_seed_start + i
                )

                for nu in viscosities:
                    # Solve Navier-Stokes equations
                    trajectory = self.ns_solver.solve(
                        omega0_field, self.T, self.dt, nu=nu, n_snapshots=self.n_snapshots
                    )

                    ns_trajectories.append(trajectory.cpu())
                    pbar.update(1)

        return torch.stack(ns_trajectories)

    def save_to_hdf5(self,
                    filename: str,
                    euler_trajectories: Optional[torch.Tensor] = None,
                    diffusion_trajectories: Optional[torch.Tensor] = None,
                    ns_trajectories: Optional[torch.Tensor] = None,
                    viscosities: Optional[List[float]] = None,
                    metadata: Optional[dict] = None):
        """
        Save trajectories to HDF5 file

        Args:
            filename: Output filename
            euler_trajectories: Euler trajectories
            diffusion_trajectories: Diffusion trajectories
            ns_trajectories: Navier-Stokes trajectories
            viscosities: List of viscosities used
            metadata: Additional metadata to save
        """
        with h5py.File(filename, 'w') as f:
            # Save trajectories
            if euler_trajectories is not None:
                f.create_dataset('euler', data=euler_trajectories.numpy(),
                               compression='gzip', compression_opts=4)

            if diffusion_trajectories is not None:
                f.create_dataset('diffusion', data=diffusion_trajectories.numpy(),
                               compression='gzip', compression_opts=4)

            if ns_trajectories is not None:
                f.create_dataset('navier_stokes', data=ns_trajectories.numpy(),
                               compression='gzip', compression_opts=4)

            # Save parameters
            params = f.create_group('parameters')
            params.attrs['N'] = self.N
            params.attrs['L'] = self.L
            params.attrs['T'] = self.T
            params.attrs['dt'] = self.dt
            params.attrs['n_snapshots'] = self.n_snapshots

            if viscosities is not None:
                params.create_dataset('viscosities', data=np.array(viscosities))

            # Save metadata
            if metadata is not None:
                meta = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, (int, float, str)):
                        meta.attrs[key] = value
                    else:
                        meta.create_dataset(key, data=value)

            print(f"Data saved to {filename}")


def generate_full_dataset(
    n_euler: int = 100,
    n_diffusion_per_euler: int = 10,
    n_ns_per_viscosity: int = 100,
    viscosity_range: tuple = (1e-4, 1e-1),
    n_viscosities: int = 10,
    output_dir: str = "data",
    N: int = 128,
    T: float = 10.0,
    dt: float = 0.01,
    n_snapshots: int = 50,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Generate complete dataset with Euler, Diffusion, and Navier-Stokes trajectories

    Args:
        n_euler: Number of Euler trajectories
        n_diffusion_per_euler: Number of diffusion trajectories per Euler trajectory
        n_ns_per_viscosity: Number of Navier-Stokes trajectories per viscosity value
        viscosity_range: (min, max) viscosity values
        n_viscosities: Number of different viscosity values to sample
        output_dir: Directory to save output files
        N: Grid resolution
        T: Simulation time
        dt: Time step
        n_snapshots: Number of snapshots per trajectory
        device: 'cuda' or 'cpu'
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Initialize generator
    generator = TrajectoryGenerator(N, L=2*np.pi, T=T, dt=dt,
                                   n_snapshots=n_snapshots, device=device)

    # Generate viscosity values (log-spaced)
    viscosities = np.logspace(np.log10(viscosity_range[0]),
                             np.log10(viscosity_range[1]),
                             n_viscosities).tolist()

    print(f"Using device: {device}")
    print(f"Viscosities: {viscosities}")
    print(f"\nGenerating dataset...")
    print(f"Target trajectories: ~{n_euler + n_euler * n_diffusion_per_euler + n_ns_per_viscosity * n_viscosities}")

    start_time = time.time()

    # 1. Generate Euler trajectories
    print("\n" + "="*50)
    print("Step 1: Generating Euler trajectories")
    print("="*50)
    euler_trajectories = generator.generate_euler_trajectories(
        n_trajectories=n_euler,
        random_seed_start=0
    )
    print(f"Euler trajectories shape: {euler_trajectories.shape}")

    # 2. Generate diffusion trajectories from Euler snapshots
    print("\n" + "="*50)
    print("Step 2: Generating diffusion trajectories from Euler snapshots")
    print("="*50)
    # Sample some snapshots to use as initial conditions
    snapshot_step = max(1, n_snapshots // n_diffusion_per_euler)
    snapshot_indices = list(range(0, n_snapshots, snapshot_step))[:n_diffusion_per_euler]
    print(f"Using snapshot indices: {snapshot_indices}")

    diffusion_trajectories = generator.generate_diffusion_from_snapshots(
        euler_trajectories=euler_trajectories,
        viscosities=viscosities,
        snapshot_indices=snapshot_indices
    )
    print(f"Diffusion trajectories shape: {diffusion_trajectories.shape}")

    # 3. Generate Navier-Stokes trajectories
    print("\n" + "="*50)
    print("Step 3: Generating Navier-Stokes trajectories")
    print("="*50)
    ns_trajectories = generator.generate_ns_trajectories(
        n_trajectories=n_ns_per_viscosity,
        viscosities=viscosities,
        random_seed_start=10000
    )
    print(f"Navier-Stokes trajectories shape: {ns_trajectories.shape}")

    elapsed = time.time() - start_time
    print(f"\nTotal generation time: {elapsed:.2f} seconds")

    # 4. Save all data
    print("\n" + "="*50)
    print("Step 4: Saving data to HDF5")
    print("="*50)
    metadata = {
        'n_euler': n_euler,
        'n_diffusion_per_euler': n_diffusion_per_euler,
        'n_ns_per_viscosity': n_ns_per_viscosity,
        'generation_time': elapsed,
        'device': device
    }

    generator.save_to_hdf5(
        filename=str(output_path / "trajectories.h5"),
        euler_trajectories=euler_trajectories,
        diffusion_trajectories=diffusion_trajectories,
        ns_trajectories=ns_trajectories,
        viscosities=viscosities,
        metadata=metadata
    )

    total_trajectories = (euler_trajectories.shape[0] +
                         diffusion_trajectories.shape[0] +
                         ns_trajectories.shape[0])

    print(f"\n{'='*50}")
    print("Dataset generation complete!")
    print(f"{'='*50}")
    print(f"Total trajectories generated: {total_trajectories}")
    print(f"  - Euler: {euler_trajectories.shape[0]}")
    print(f"  - Diffusion: {diffusion_trajectories.shape[0]}")
    print(f"  - Navier-Stokes: {ns_trajectories.shape[0]}")
    print(f"Output: {output_path / 'trajectories.h5'}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average time per trajectory: {elapsed/total_trajectories:.3f} seconds")
