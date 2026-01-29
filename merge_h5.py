#!/usr/bin/env python
"""
Merge HDF5 files from parallel generation into a single file.

Reorganizes data so viscosities are contiguous:
    - Diffusion: [all visc0][all visc1]...[all visc_m]
    - Navier-Stokes: [all visc0][all visc1]...[all visc_m]

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

    m_visc = len(viscosities)

    # Calculate total sizes and per-GPU counts
    total_euler = 0
    total_diff = 0
    total_ns = 0
    gpu_info = []

    for gpu_id in range(num_gpus):
        with h5py.File(input_path / f"trajectories_gpu{gpu_id}.h5", "r") as f:
            info = {
                "euler_count": f.attrs["euler_count"],
                "diff_count": f.attrs["diff_count"],
                "ns_count": f.attrs["ns_count"],
            }
            gpu_info.append(info)
            total_euler += info["euler_count"]
            total_diff += info["diff_count"]
            total_ns += info["ns_count"]

    print(f"Merging {num_gpus} files into {output_file}")
    print(f"  Euler: {total_euler}")
    print(f"  Diffusion: {total_diff} ({total_diff // m_visc} per viscosity)")
    print(f"  Navier-Stokes: {total_ns} ({total_ns // m_visc} per viscosity)")

    # Calculate samples per viscosity
    n_diff_per_visc = total_diff // m_visc
    n_ns_per_visc = total_ns // m_visc

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

        chunk_size = 64

        # --- Euler: simple concatenation (no viscosity) ---
        euler_offset = 0
        for gpu_id in tqdm(range(num_gpus), desc="Merging Euler"):
            gpu_file = input_path / f"trajectories_gpu{gpu_id}.h5"
            with h5py.File(gpu_file, "r") as f:
                euler_count = gpu_info[gpu_id]["euler_count"]
                for i in range(0, euler_count, chunk_size):
                    n = min(chunk_size, euler_count - i)
                    out["euler"][euler_offset + i : euler_offset + i + n] = f["euler"][i : i + n]
                euler_offset += euler_count

        # --- Diffusion: reorganize by viscosity ---
        # Each GPU file has [visc0][visc1]...[visc_m] stored consecutively
        # We want output: [all_visc0][all_visc1]...[all_visc_m]
        visc_offsets_diff = [v_idx * n_diff_per_visc for v_idx in range(m_visc)]

        for gpu_id in tqdm(range(num_gpus), desc="Merging Diffusion"):
            gpu_file = input_path / f"trajectories_gpu{gpu_id}.h5"
            with h5py.File(gpu_file, "r") as f:
                diff_count = gpu_info[gpu_id]["diff_count"]
                n_per_visc_this_gpu = diff_count // m_visc

                for v_idx in range(m_visc):
                    src_start = v_idx * n_per_visc_this_gpu
                    dst_start = visc_offsets_diff[v_idx]

                    for i in range(0, n_per_visc_this_gpu, chunk_size):
                        n = min(chunk_size, n_per_visc_this_gpu - i)
                        out["diffusion"][dst_start + i : dst_start + i + n] = \
                            f["diffusion"][src_start + i : src_start + i + n]

                    visc_offsets_diff[v_idx] += n_per_visc_this_gpu

        # --- Navier-Stokes: reorganize by viscosity ---
        # NS uses global structure: v_idx = global_idx // N_ns_ics
        # Each GPU handles a slice [ns_start, ns_end) of global indices
        # N_ns_ics = total_ns // m_visc (samples per viscosity)
        N_ns_ics = total_ns // m_visc
        visc_offsets_ns = [v_idx * N_ns_ics for v_idx in range(m_visc)]

        for gpu_id in tqdm(range(num_gpus), desc="Merging NS"):
            gpu_file = input_path / f"trajectories_gpu{gpu_id}.h5"
            with h5py.File(gpu_file, "r") as f:
                ns_start = f.attrs["ns_start"]
                ns_count = gpu_info[gpu_id]["ns_count"]

                # Process each sample, mapping global_idx to viscosity
                local_idx = 0
                while local_idx < ns_count:
                    global_idx = ns_start + local_idx
                    v_idx = global_idx // N_ns_ics
                    ic_idx = global_idx % N_ns_ics

                    # Find how many consecutive samples share this viscosity
                    remaining_in_visc = N_ns_ics - ic_idx
                    remaining_in_gpu = ns_count - local_idx
                    batch_n = min(chunk_size, remaining_in_visc, remaining_in_gpu)

                    dst_start = visc_offsets_ns[v_idx]
                    out["navier_stokes"][dst_start : dst_start + batch_n] = \
                        f["navier_stokes"][local_idx : local_idx + batch_n]

                    visc_offsets_ns[v_idx] += batch_n
                    local_idx += batch_n

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
