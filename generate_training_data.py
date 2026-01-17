#!/usr/bin/env python3
"""
Optimized training data generation using joblib for better PyTorch compatibility:
- 64 Euler trajectories
- 64 Diffusion trajectories (viscosity=1e-3)  
- 64 Navier-Stokes trajectories
"""

import argparse
import torch
import time
import numpy as np
from pathlib import Path
from data_generator import TrajectoryGenerator
from joblib import Parallel, delayed


def generate_batch_euler_trajectories(start_idx, batch_size, N, L, n_snapshots, T, dt, random_seed_start):
    """Worker function for parallel Euler trajectory generation (batched)"""
    
    # Force CPU for multiprocessing
    device = 'cpu'
    
    # Initialize generator once per batch
    generator = TrajectoryGenerator(
        N=N,
        L=L,
        T=T,
        dt=dt,
        n_snapshots=n_snapshots,
        device=device
    )
    
    # Generate trajectories
    trajectories = []
    for i in range(batch_size):
        # Generate random initial condition
        omega0 = generator.generate_random_initial_condition(
            omega0=1.0, n_modes=5, seed=random_seed_start + i
        )
        
        # Solve Euler equations
        trajectory = generator.euler_solver.solve(
            omega0, T, dt, nu=0.0, n_snapshots=n_snapshots
        )
        trajectories.append(trajectory.cpu())
    
    return start_idx, torch.stack(trajectories)


def generate_batch_diffusion_trajectories(start_idx, batch_size, N, L, n_snapshots, T, dt, viscosity, random_seed_start, euler_trajectories):
    """Worker function for parallel diffusion trajectory generation (batched)"""
    
    # Force CPU for multiprocessing
    device = 'cpu'
    
    # Initialize generator once per batch
    generator = TrajectoryGenerator(
        N=N,
        L=L,
        T=T,
        dt=dt,
        n_snapshots=n_snapshots,
        device=device
    )
    
    # Set random seed for reproducible sampling
    np.random.seed(random_seed_start)
    
    trajectories = []
    for i in range(batch_size):
        # Randomly sample from Euler trajectories
        traj_idx = np.random.randint(0, euler_trajectories.shape[0])
        time_idx = np.random.randint(0, euler_trajectories.shape[1])

        #print(f"Sampling Euler traj {traj_idx}, time {time_idx} for diffusion traj {start_idx + i}")
        #omega0 = euler_trajectories[traj_idx, time_idx].to(device)
        omega0 = generator.generate_random_initial_condition(
            omega0=1.0, n_modes=5, seed=random_seed_start + i
        )
        
        # Solve diffusion equation
        trajectory = generator.diffusion_solver.solve(
            omega0, T, dt, nu=viscosity, n_snapshots=n_snapshots
        )
        trajectories.append(trajectory.cpu())
    
    return start_idx, torch.stack(trajectories)


def generate_batch_ns_trajectories(start_idx, batch_size, N, L, n_snapshots, T, dt, viscosity, random_seed_start):
    """Worker function for parallel Navier-Stokes trajectory generation (batched)"""
    
    # Force CPU for multiprocessing
    device = 'cpu'
    
    # Initialize generator once per batch
    generator = TrajectoryGenerator(
        N=N,
        L=L,
        T=T,
        dt=dt,
        n_snapshots=n_snapshots,
        device=device
    )
    
    trajectories = []
    for i in range(batch_size):
        # Generate random initial condition
        omega0 = generator.generate_random_initial_condition(
            omega0=1.0, n_modes=5, seed=random_seed_start + i
        )
        
        # Solve Navier-Stokes equation
        trajectory = generator.ns_solver.solve(
            omega0, T, dt, nu=viscosity, n_snapshots=n_snapshots
        )
        trajectories.append(trajectory.cpu())
    
    return start_idx, torch.stack(trajectories)


def main():
    parser = argparse.ArgumentParser(description='Generate training dataset with fixed parameters')
    parser.add_argument('--output-dir', type=str, default='training_data4',
                       help='Output directory (default: training_data)')
    parser.add_argument('--resolution', type=int, default=128,
                       help='Grid resolution N (default: 128)')
    parser.add_argument('--time', type=float, default=4.0,
                       help='Simulation time per trajectory (default: 4.0)')
    parser.add_argument('--dt', type=float, default=0.001,
                       help='Time step (default: 0.001)')
    parser.add_argument('--snapshots', type=int, default=100,
                       help='Number of snapshots per trajectory (default: 100)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (default: auto)')
    parser.add_argument('--cores', type=int, default=None,
                       help='Number of CPU cores to use for parallel processing (default: all available)')
    
    args = parser.parse_args()
    
    # Fixed parameters for this generation
    N_EULER = 16
    N_DIFFUSION = 2048  
    N_NS = 16
    VISCOSITY = 1e-2
    
    # Force CPU for multiprocessing
    device = 'cpu'
    
    # Determine number of cores - use -1 for joblib (auto-detect)
    n_cores = args.cores if args.cores else -1
    
    # Calculate optimal batch size: aim for 4-8 trajectories per worker
    # With joblib, we can be more flexible
    if n_cores == -1:
        # Let joblib decide, but aim for reasonable batch sizes
        import os
        actual_cores = os.cpu_count()
        min_trajectories_per_worker = 4
        optimal_batch_size = max(min_trajectories_per_worker, N_EULER // min(actual_cores, 16))
    else:
        min_trajectories_per_worker = 4
        optimal_batch_size = max(min_trajectories_per_worker, N_EULER // n_cores)
    
    print("="*60)
    print("Training Data Generation (using joblib)")
    print("="*60)
    print(f"Euler trajectories: {N_EULER}")
    print(f"Diffusion trajectories: {N_DIFFUSION} (viscosity={VISCOSITY})")
    print(f"Navier-Stokes trajectories: {N_NS}")
    print(f"Grid resolution: {args.resolution}×{args.resolution}")
    print(f"Snapshots per trajectory: {args.snapshots}")
    print(f"Simulation time: {args.time}")
    print(f"Time step: {args.dt}")
    print(f"Device: {device}")
    print(f"Cores: {n_cores} (-1 = auto)")
    print(f"Batch size per worker: ~{optimal_batch_size}")
    print("="*60)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize generator (just for saving, not for parallel work)
    generator = TrajectoryGenerator(
        N=args.resolution,
        L=2*np.pi,
        T=args.time,
        dt=args.dt,
        n_snapshots=args.snapshots,
        device=device
    )
    
    total_start_time = time.time()
    
    # 1. Generate Euler trajectories using joblib parallel processing
    print("\n" + "="*50)
    print(f"Step 1: Generating Euler trajectories")
    print("="*50)
    start_time = time.time()
    
    # Prepare arguments for parallel processing
    euler_args = []
    for i in range(0, N_EULER, optimal_batch_size):
        actual_batch = min(optimal_batch_size, N_EULER - i)
        euler_args.append((i, actual_batch, args.resolution, 2*np.pi, args.snapshots, args.time, args.dt, i))
    
    print(f"Processing {len(euler_args)} batches in parallel...")
    
    # Generate trajectories in parallel using joblib
    euler_results = Parallel(n_jobs=n_cores, verbose=10)(
        delayed(generate_batch_euler_trajectories)(*euler_arg)
        for euler_arg in euler_args
    )
    
    # Collect and sort results by start_idx
    euler_results.sort(key=lambda x: x[0])
    euler_trajectories = []
    for start_idx, batch_trajectories in euler_results:
        for i in range(batch_trajectories.shape[0]):
            euler_trajectories.append(batch_trajectories[i])
    euler_trajectories = torch.stack(euler_trajectories[:N_EULER])
    
    euler_time = time.time() - start_time
    print(f"✓ Euler generation complete: {euler_time:.2f} seconds")
    print(f"  Shape: {euler_trajectories.shape}")
    
    # Save Euler data immediately to free memory
    print("  Saving to disk...")
    generator.save_to_hdf5(
        filename=str(output_path / "euler_trajectories.h5"),
        euler_trajectories=euler_trajectories,
        metadata={'type': 'euler', 'n_trajectories': N_EULER}
    )
    print("  ✓ Saved")
    del euler_trajectories  # Free memory
    
    # 2. Generate diffusion trajectories with fixed viscosity using joblib parallel processing
    print("\n" + "="*50)
    print(f"Step 2: Generating diffusion trajectories")
    print("="*50)
    start_time = time.time()
    
    # Load Euler trajectories for sampling
    print("  Loading Euler trajectories for sampling...")
    import h5py
    with h5py.File(output_path / "euler_trajectories.h5", 'r') as f:
        euler_trajectories = torch.from_numpy(f['euler'][:])
    print(f"  Loaded Euler trajectories: {euler_trajectories.shape}")
    
    # Prepare arguments for parallel processing
    diffusion_args = []
    for i in range(0, N_DIFFUSION, optimal_batch_size):
        actual_batch = min(optimal_batch_size, N_DIFFUSION - i)
        diffusion_args.append((i, actual_batch, args.resolution, 2*np.pi, args.snapshots, args.time, args.dt, VISCOSITY, 10000 + i, euler_trajectories))
    
    print(f"Processing {len(diffusion_args)} batches in parallel...")
    
    # Generate trajectories in parallel using joblib
    diffusion_results = Parallel(n_jobs=n_cores, verbose=10)(
        delayed(generate_batch_diffusion_trajectories)(*diffusion_arg)
        for diffusion_arg in diffusion_args
    )
    
    # Collect and sort results
    diffusion_results.sort(key=lambda x: x[0])
    diffusion_trajectories = []
    for start_idx, batch_trajectories in diffusion_results:
        for i in range(batch_trajectories.shape[0]):
            diffusion_trajectories.append(batch_trajectories[i])
    diffusion_trajectories = torch.stack(diffusion_trajectories[:N_DIFFUSION])
    
    diffusion_time = time.time() - start_time
    print(f"✓ Diffusion generation complete: {diffusion_time:.2f} seconds")
    print(f"  Shape: {diffusion_trajectories.shape}")
    
    # Save diffusion data
    print("  Saving to disk...")
    generator.save_to_hdf5(
        filename=str(output_path / "diffusion_trajectories.h5"),
        diffusion_trajectories=diffusion_trajectories,
        viscosities=[VISCOSITY],
        metadata={'type': 'diffusion', 'viscosity': VISCOSITY, 'n_trajectories': N_DIFFUSION}
    )
    print("  ✓ Saved")
    del diffusion_trajectories  # Free memory
    
    # 3. Generate Navier-Stokes trajectories using joblib parallel processing
    print("\n" + "="*50)
    print(f"Step 3: Generating Navier-Stokes trajectories")
    print("="*50)
    start_time = time.time()
    
    # Prepare arguments for parallel processing
    ns_args = []
    for i in range(0, N_NS, optimal_batch_size):
        actual_batch = min(optimal_batch_size, N_NS - i)
        ns_args.append((i, actual_batch, args.resolution, 2*np.pi, args.snapshots, args.time, args.dt, VISCOSITY, 20000 + i))
    
    print(f"Processing {len(ns_args)} batches in parallel...")
    
    # Generate trajectories in parallel using joblib
    ns_results = Parallel(n_jobs=n_cores, verbose=10)(
        delayed(generate_batch_ns_trajectories)(*ns_arg)
        for ns_arg in ns_args
    )
    
    # Collect and sort results
    ns_results.sort(key=lambda x: x[0])
    ns_trajectories = []
    for start_idx, batch_trajectories in ns_results:
        for i in range(batch_trajectories.shape[0]):
            ns_trajectories.append(batch_trajectories[i])
    ns_trajectories = torch.stack(ns_trajectories[:N_NS])
    
    ns_time = time.time() - start_time
    print(f"✓ Navier-Stokes generation complete: {ns_time:.2f} seconds")
    print(f"  Shape: {ns_trajectories.shape}")
    
    # Save N-S data
    print("  Saving to disk...")
    generator.save_to_hdf5(
        filename=str(output_path / "navier_stokes_trajectories.h5"),
        ns_trajectories=ns_trajectories,
        viscosities=[VISCOSITY],
        metadata={'type': 'navier_stokes', 'viscosity': VISCOSITY, 'n_trajectories': N_NS}
    )
    print("  ✓ Saved")
    del ns_trajectories  # Free memory
    
    total_time = time.time() - total_start_time
    total_trajectories = N_EULER + N_DIFFUSION + N_NS
    
    print(f"\n{'='*60}")
    print("✓ Training data generation complete!")
    print(f"{'='*60}")
    print(f"Total trajectories generated: {total_trajectories}")
    print(f"  - Euler: {N_EULER}")
    print(f"  - Diffusion: {N_DIFFUSION}")
    print(f"  - Navier-Stokes: {N_NS}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average time per trajectory: {total_time/total_trajectories:.3f} seconds")
    print(f"\nFiles created:")
    print(f"  - euler_trajectories.h5")
    print(f"  - diffusion_trajectories.h5") 
    print(f"  - navier_stokes_trajectories.h5")


if __name__ == '__main__':
    main()