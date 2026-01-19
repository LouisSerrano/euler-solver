#!/usr/bin/env python
"""
Merge HDF5 files from parallel generation into a single file.

Usage:
    python merge_h5.py --input_dir /path/to/output --num_gpus 4
"""
import h5py
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


def merge_files(input_dir, num_gpus, output_name="trajectories_train.h5"):
    input_path = Path(input_dir)
    output_file = input_path / output_name
    
    # Read metadata from first file
    with h5py.File(input_path / "trajectories_gpu0.h5", "r") as f0:
        viscosities = f0.attrs["viscosities"]
        sim_res = f0.attrs["sim_res"]
        save_res = f0.attrs["save_res"]
        T = f0.attrs["T"]
        n_snapshots = f0.attrs["n_snapshots"]
    
    # Calculate total sizes
    total_euler = 0
    total_diff = 0
    total_ns = 0
    
    for gpu_id in range(num_gpus):
        with h5py.File(input_path / f"trajectories_gpu{gpu_id}.h5", "r") as f:
            total_euler += f.attrs["euler_count"]
            total_diff += f.attrs["diff_count"]
            total_ns += f.attrs["ns_count"]
    
    print(f"Merging {num_gpus} files into {output_file}")
    print(f"  Euler: {total_euler}")
    print(f"  Diffusion: {total_diff}")
    print(f"  Navier-Stokes: {total_ns}")
    
    with h5py.File(output_file, "w") as out:
        # Create output datasets
        out.create_dataset("euler", (total_euler, n_snapshots, save_res, save_res),
                          dtype="f4", compression="lzf",
                          chunks=(1, n_snapshots, save_res, save_res))
        out.create_dataset("diffusion", (total_diff, n_snapshots, save_res, save_res),
                          dtype="f4", compression="lzf",
                          chunks=(1, n_snapshots, save_res, save_res))
        out.create_dataset("navier_stokes", (total_ns, n_snapshots, save_res, save_res),
                          dtype="f4", compression="lzf",
                          chunks=(1, n_snapshots, save_res, save_res))
        
        out.attrs["viscosities"] = viscosities
        out.attrs["sim_res"] = sim_res
        out.attrs["save_res"] = save_res
        out.attrs["T"] = T
        out.attrs["n_snapshots"] = n_snapshots
        
        # Copy data from each GPU file
        for gpu_id in tqdm(range(num_gpus), desc="Merging GPUs"):
            gpu_file = input_path / f"trajectories_gpu{gpu_id}.h5"
            with h5py.File(gpu_file, "r") as f:
                euler_start = f.attrs["euler_start"]
                euler_count = f.attrs["euler_count"]
                diff_start = f.attrs["diff_start"]
                diff_count = f.attrs["diff_count"]
                ns_start = f.attrs["ns_start"]
                ns_count = f.attrs["ns_count"]
                
                # Copy in chunks to avoid memory issues
                chunk_size = 64
                
                for i in range(0, euler_count, chunk_size):
                    n = min(chunk_size, euler_count - i)
                    out["euler"][euler_start + i : euler_start + i + n] = f["euler"][i : i + n]
                
                for i in range(0, diff_count, chunk_size):
                    n = min(chunk_size, diff_count - i)
                    out["diffusion"][diff_start + i : diff_start + i + n] = f["diffusion"][i : i + n]
                
                for i in range(0, ns_count, chunk_size):
                    n = min(chunk_size, ns_count - i)
                    out["navier_stokes"][ns_start + i : ns_start + i + n] = f["navier_stokes"][i : i + n]
    
    print(f"Done! Merged file: {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge parallel HDF5 files")
    parser.add_argument("--input_dir", type=str, default="/mnt/home/lserrano/ceph/data/euler_ns/",
                       help="Directory with GPU files")
    parser.add_argument("--num_gpus", type=int, required=True, help="Number of GPU files to merge")
    parser.add_argument("--output_name", type=str, default="trajectories_train.h5", help="Output filename")
    
    args = parser.parse_args()
    merge_files(args.input_dir, args.num_gpus, args.output_name)
