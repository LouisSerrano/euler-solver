import torch
import numpy as np
import h5py
import os
import time
from pathlib import Path
from tqdm import tqdm

from solver_2d_ns_euler_midpoint import VorticitySpectral2D, DiffusionSpectral2D, downsample_spectral

def save_batch_to_h5(f, group_name, data, start_idx):
    """Save a batch of data to an HDF5 dataset."""
    B = data.shape[0]
    f[group_name][start_idx:start_idx+B] = data.cpu().numpy()

def generate_dataset(
    N_total=8192,
    N_ns_ics=256,
    m_visc=16,
    T=10.0,
    n_snapshots=50,
    sim_res=512,
    save_res=256,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir="/mnt/home/lserrano/ceph/data/euler_ns/",
    filename="trajectories_train.h5",
    batch_size=64,
    seed_offset=0 # for train/test split
):
    print("device:", device)
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / filename
    
    viscosities = np.logspace(-4, -2, m_visc)
    print(f"Viscosities: {viscosities}")
    
    solver = VorticitySpectral2D(N=sim_res, device=device)
    diff_solver = DiffusionSpectral2D(N=sim_res, device=device)
    
    # Use torch.no_grad() for the entire generation process
    with torch.no_grad():
        # 1. Euler Trajectories
        print(f"\n--- Generating {N_total} Euler Trajectories ---")
        with h5py.File(filename, "w") as f:
            f.create_dataset("euler", (N_total, n_snapshots, save_res, save_res), 
                         dtype="f4", compression="lzf",
                         chunks=(1, n_snapshots, save_res, save_res))
            
            all_euler_snapshots = [] # We'll store a few for diffusion ICs
            
            for i in tqdm(range(0, N_total, batch_size)):
                curr_batch = min(batch_size, N_total - i)
                
                # A. IC Generation
                t0 = time.perf_counter()
                omega0 = solver.batch_random_smooth_init(B=curr_batch, seed=i + seed_offset)
                torch.cuda.synchronize()
                t_ic = time.perf_counter() - t0
                
                # B. Solve Euler (nu=0)
                t0 = time.perf_counter()
                _, traj, _, _ = solver.solve(omega0, T=T, n_snapshots=n_snapshots, nu=0.0, quiet=False, stepper="midpoint")
                torch.cuda.synchronize()
                t_solve = time.perf_counter() - t0
                
                # C. Batched downsample
                t0 = time.perf_counter()
                traj_down = downsample_spectral(traj, save_res)
                torch.cuda.synchronize()
                t_down = time.perf_counter() - t0
                
                # D. HDF5 IO
                t0 = time.perf_counter()
                save_batch_to_h5(f, "euler", traj_down, i)
                t_io = time.perf_counter() - t0
                
                print(f"Batch {i//batch_size} | IC: {t_ic:.2f}s | Solve: {t_solve:.2f}s | Down: {t_down:.2f}s | IO: {t_io:.2f}s")
                
                # Get random development snapshots for diffusion pool
                t_indices = torch.randint(0, n_snapshots, (curr_batch,), device=device)
                random_snaps = traj[torch.arange(curr_batch), t_indices].clone()
                all_euler_snapshots.append(random_snaps)

            # Build diffusion IC pool
            diffusion_ic_pool = torch.cat(all_euler_snapshots, dim=0)
            shuffle_idx = torch.randperm(N_total, device=device)
            diffusion_ic_pool = diffusion_ic_pool[shuffle_idx]
            
            # 2. Diffusion Trajectories
            print(f"\n--- Generating {N_total} Diffusion Trajectories ---")
            n_diff_per_v = N_total // m_visc
            f.create_dataset("diffusion", (N_total, n_snapshots, save_res, save_res), 
                             dtype="f4", compression="lzf",
                             chunks=(1, n_snapshots, save_res, save_res))
            
            for v_idx, nu in enumerate(viscosities):
                print(f"  Diffusion nu={nu:.2e}")
                for i in tqdm(range(0, n_diff_per_v, batch_size)):
                    curr_batch = min(batch_size, n_diff_per_v - i)
                    idx_start = v_idx * n_diff_per_v + i
                    
                    omega0 = diffusion_ic_pool[idx_start : idx_start + curr_batch]
                    
                    # Solve Diffusion (Vectorized)
                    t0 = time.perf_counter()
                    _, traj, _, _ = diff_solver.solve(omega0, T=T, n_snapshots=n_snapshots, nu=nu)
                    torch.cuda.synchronize()
                    t_solve = time.perf_counter() - t0
                    
                    # Batched downsample
                    t0 = time.perf_counter()
                    traj_down = downsample_spectral(traj, save_res)
                    torch.cuda.synchronize()
                    t_down = time.perf_counter() - t0
                    
                    # IO
                    t0 = time.perf_counter()
                    save_batch_to_h5(f, "diffusion", traj_down, idx_start)
                    t_io = time.perf_counter() - t0
                    
                    if i == 0 and v_idx == 0:
                        print(f"\n[Diff Batch 0] Solve: {t_solve:.2f}s | Down: {t_down:.2f}s | IO: {t_io:.2f}s")

            # 3. Navier-Stokes Trajectories
            print(f"\n--- Generating {N_ns_ics * m_visc} Navier-Stokes Trajectories ---")
            f.create_dataset("navier_stokes", (N_ns_ics * m_visc, n_snapshots, save_res, save_res), 
                             dtype="f4", compression="lzf",
                             chunks=(1, n_snapshots, save_res, save_res))
            
            # New ICs for NS
            ns_ics = solver.batch_random_smooth_init(B=N_ns_ics, seed=20000 + seed_offset)
            
            for v_idx, nu in enumerate(viscosities):
                print(f"  Navier-Stokes nu={nu:.2e}")
                for i in tqdm(range(0, N_ns_ics, batch_size)):
                    curr_batch = min(batch_size, N_ns_ics - i)
                    idx_start = v_idx * N_ns_ics + i
                    
                    omega0 = ns_ics[i : i + curr_batch]
                    
                    # Solve NS
                    t0 = time.perf_counter()
                    _, traj, _, _ = solver.solve(omega0, T=T, n_snapshots=n_snapshots, nu=nu, quiet=False, stepper="midpoint")
                    torch.cuda.synchronize()
                    t_solve = time.perf_counter() - t0
                    
                    # Batched downsample
                    t0 = time.perf_counter()
                    traj_down = downsample_spectral(traj, save_res)
                    torch.cuda.synchronize()
                    t_down = time.perf_counter() - t0
                    
                    # IO
                    t0 = time.perf_counter()
                    save_batch_to_h5(f, "navier_stokes", traj_down, idx_start)
                    t_io = time.perf_counter() - t0
                    
                    print(f"Visc {v_idx} Batch {i//batch_size} | Solve: {t_solve:.2f}s | Down: {t_down:.2f}s | IO: {t_io:.2f}s")

            # Save metadata
            f.attrs["viscosities"] = viscosities
            f.attrs["sim_res"] = sim_res
            f.attrs["save_res"] = save_res
            f.attrs["T"] = T
            f.attrs["n_snapshots"] = n_snapshots

    print(f"\nDone! Dataset saved to {filename}")

if __name__ == "__main__":
    # Full Run Configuration
    generate_dataset(
        N_total=32, #8192, 
        N_ns_ics=32, #512, 
        m_visc=1,#16, 
        T=10.0, 
        n_snapshots=50, 
        batch_size=32, # Balanced for 512x512 on most GPUs
        output_dir="/mnt/home/lserrano/ceph/data/euler_ns/"
    )
