#!/usr/bin/env python
"""
Parallel Data Generator - Generates a slice of trajectories for multi-GPU execution.

Usage:
    python data_generator_parallel.py --gpu_id 0 --num_gpus 4 --output_dir /path/to/output
    
This will generate trajectories [0, N/4), [N/4, N/2), etc. based on gpu_id.
Run merge_h5.py after all jobs complete to combine into a single file.
"""
import torch
import numpy as np
import h5py
import os
import time
import argparse
from pathlib import Path
from tqdm import tqdm

from solver_2d_ns_euler_midpoint import VorticitySpectral2D, DiffusionSpectral2D, downsample_spectral


def save_batch_to_h5(f, group_name, data, start_idx):
    """Save a batch of data to an HDF5 dataset."""
    B = data.shape[0]
    f[group_name][start_idx:start_idx+B] = data.cpu().numpy()


def generate_slice(
    gpu_id,
    num_gpus,
    N_total=8192,
    N_ns_ics=512,
    m_visc=16,
    T=10.0,
    n_snapshots=50,
    sim_res=512,
    save_res=256,
    output_dir="/mnt/home/lserrano/ceph/data/euler_ns/",
    batch_size=32,
    seed_offset=0
):
    device = f"cuda:{gpu_id % torch.cuda.device_count()}"
    print(f"[GPU {gpu_id}] Using device: {device}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Each GPU writes to its own file
    filename = output_path / f"trajectories_gpu{gpu_id}.h5"
    
    viscosities = np.logspace(-4, -2, m_visc)
    
    solver = VorticitySpectral2D(N=sim_res, device=device)
    diff_solver = DiffusionSpectral2D(N=sim_res, device=device)
    
    # Calculate this GPU's slice of work
    euler_per_gpu = N_total // num_gpus
    euler_start = gpu_id * euler_per_gpu
    euler_end = euler_start + euler_per_gpu if gpu_id < num_gpus - 1 else N_total
    euler_count = euler_end - euler_start
    
    diff_per_gpu = N_total // num_gpus
    diff_start = gpu_id * diff_per_gpu
    diff_end = diff_start + diff_per_gpu if gpu_id < num_gpus - 1 else N_total
    diff_count = diff_end - diff_start
    
    ns_per_gpu = (N_ns_ics * m_visc) // num_gpus
    ns_start = gpu_id * ns_per_gpu
    ns_end = ns_start + ns_per_gpu if gpu_id < num_gpus - 1 else N_ns_ics * m_visc
    ns_count = ns_end - ns_start
    
    print(f"[GPU {gpu_id}] Euler: {euler_start}-{euler_end} ({euler_count})")
    print(f"[GPU {gpu_id}] Diffusion: {diff_start}-{diff_end} ({diff_count})")
    print(f"[GPU {gpu_id}] NS: {ns_start}-{ns_end} ({ns_count})")
    
    with torch.no_grad():
        with h5py.File(filename, "w") as f:
            # Store slice info for merging
            f.attrs["gpu_id"] = gpu_id
            f.attrs["num_gpus"] = num_gpus
            f.attrs["euler_start"] = euler_start
            f.attrs["euler_count"] = euler_count
            f.attrs["diff_start"] = diff_start
            f.attrs["diff_count"] = diff_count
            f.attrs["ns_start"] = ns_start
            f.attrs["ns_count"] = ns_count
            f.attrs["viscosities"] = viscosities
            f.attrs["sim_res"] = sim_res
            f.attrs["save_res"] = save_res
            f.attrs["T"] = T
            f.attrs["n_snapshots"] = n_snapshots
            
            # Create datasets for this GPU's slice
            f.create_dataset("euler", (euler_count, n_snapshots, save_res, save_res),
                           dtype="f4", compression="lzf",
                           chunks=(1, n_snapshots, save_res, save_res))
            
            f.create_dataset("diffusion", (diff_count, n_snapshots, save_res, save_res),
                           dtype="f4", compression="lzf",
                           chunks=(1, n_snapshots, save_res, save_res))
            
            f.create_dataset("navier_stokes", (ns_count, n_snapshots, save_res, save_res),
                           dtype="f4", compression="lzf",
                           chunks=(1, n_snapshots, save_res, save_res))
            
            all_euler_snapshots = []
            
            # 1. Euler Trajectories
            print(f"\n[GPU {gpu_id}] --- Generating {euler_count} Euler Trajectories ---")
            local_idx = 0
            for i in tqdm(range(euler_start, euler_end, batch_size), desc=f"GPU{gpu_id} Euler"):
                curr_batch = min(batch_size, euler_end - i)
                
                t0 = time.perf_counter()
                omega0 = solver.batch_random_smooth_init(B=curr_batch, seed=i + seed_offset)
                torch.cuda.synchronize()
                t_ic = time.perf_counter() - t0
                
                t0 = time.perf_counter()
                _, traj, _, _ = solver.solve(omega0, T=T, n_snapshots=n_snapshots, nu=0.0, quiet=True, stepper="midpoint")
                torch.cuda.synchronize()
                t_solve = time.perf_counter() - t0
                
                t0 = time.perf_counter()
                traj_down = downsample_spectral(traj, save_res)
                torch.cuda.synchronize()
                t_down = time.perf_counter() - t0
                
                t0 = time.perf_counter()
                save_batch_to_h5(f, "euler", traj_down, local_idx)
                t_io = time.perf_counter() - t0
                
                local_idx += curr_batch
                
                # Sample for diffusion ICs
                t_indices = torch.randint(0, n_snapshots, (curr_batch,), device=device)
                random_snaps = traj[torch.arange(curr_batch), t_indices].clone()
                all_euler_snapshots.append(random_snaps)
                
                if local_idx <= batch_size:
                    print(f"[GPU {gpu_id}] IC: {t_ic:.2f}s | Solve: {t_solve:.2f}s | Down: {t_down:.2f}s | IO: {t_io:.2f}s")
            
            # 2. Diffusion Trajectories
            print(f"\n[GPU {gpu_id}] --- Generating {diff_count} Diffusion Trajectories ---")
            diffusion_ic_pool = torch.cat(all_euler_snapshots, dim=0)
            shuffle_idx = torch.randperm(len(diffusion_ic_pool), device=device)
            diffusion_ic_pool = diffusion_ic_pool[shuffle_idx]
            
            n_diff_per_v = diff_count // m_visc
            local_idx = 0
            
            for v_idx, nu in enumerate(viscosities):
                for i in tqdm(range(0, n_diff_per_v, batch_size), desc=f"GPU{gpu_id} Diff nu={nu:.0e}", leave=False):
                    curr_batch = min(batch_size, n_diff_per_v - i)
                    pool_idx = v_idx * n_diff_per_v + i
                    
                    omega0 = diffusion_ic_pool[pool_idx : pool_idx + curr_batch]
                    
                    _, traj, _, _ = diff_solver.solve(omega0, T=T, n_snapshots=n_snapshots, nu=nu)
                    torch.cuda.synchronize()
                    
                    traj_down = downsample_spectral(traj, save_res)
                    save_batch_to_h5(f, "diffusion", traj_down, local_idx)
                    local_idx += curr_batch
            
            # 3. Navier-Stokes Trajectories
            print(f"\n[GPU {gpu_id}] --- Generating {ns_count} Navier-Stokes Trajectories ---")
            
            # Determine which viscosities and ICs this GPU handles
            ns_ics_total = N_ns_ics * m_visc
            local_idx = 0
            
            for global_idx in tqdm(range(ns_start, ns_end, batch_size), desc=f"GPU{gpu_id} NS"):
                curr_batch = min(batch_size, ns_end - global_idx)
                
                # Decode v_idx and ic_idx from global_idx
                v_idx = global_idx // N_ns_ics
                ic_start = global_idx % N_ns_ics
                nu = viscosities[v_idx]
                
                omega0 = solver.batch_random_smooth_init(B=curr_batch, seed=20000 + seed_offset + global_idx)
                
                _, traj, _, _ = solver.solve(omega0, T=T, n_snapshots=n_snapshots, nu=nu, quiet=True, stepper="midpoint")
                torch.cuda.synchronize()
                
                traj_down = downsample_spectral(traj, save_res)
                save_batch_to_h5(f, "navier_stokes", traj_down, local_idx)
                local_idx += curr_batch
    
    print(f"\n[GPU {gpu_id}] Done! Saved to {filename}")
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel trajectory generation")
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID (0-indexed)")
    parser.add_argument("--num_gpus", type=int, required=True, help="Total number of GPUs")
    parser.add_argument("--output_dir", type=str, default="/mnt/home/lserrano/ceph/data/euler_ns/",
                       help="Output directory")
    parser.add_argument("--N_total", type=int, default=8192, help="Total Euler trajectories")
    parser.add_argument("--N_ns_ics", type=int, default=512, help="NS ICs per viscosity")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    generate_slice(
        gpu_id=args.gpu_id,
        num_gpus=args.num_gpus,
        N_total=args.N_total,
        N_ns_ics=args.N_ns_ics,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
