#!/usr/bin/env python3
"""
Parametric viscosity experiment script for comparing Euler vs Navier-Stokes dynamics.
Generates trajectories for N initial conditions across a range of viscosities and
different initial condition types, then saves data and creates comparison plots.

Usage:
    python viscosity_experiment.py --n-initial 10 --viscosities 0.0,1e-4,1e-3,1e-2 --init-types random,vortex
"""

import argparse
import torch
import torch.fft as fft
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from data_generator import TrajectoryGenerator
from joblib import Parallel, delayed
import h5py
from typing import List, Tuple, Dict


class ViscosityExperiment:
    """
    Manages viscosity experiments comparing Euler vs Navier-Stokes dynamics
    across different initial conditions and viscosity values.
    """
    
    def __init__(self, N: int = 256, L: float = 2*np.pi, T: float = 4.0, 
                 dt: float = 0.001, n_snapshots: int = 100, device: str = 'cpu',
                 filter_order: int = 16):
        self.N = N
        self.L = L
        self.T = T
        self.dt = dt
        self.n_snapshots = n_snapshots
        self.device = device
        
        # Initialize trajectory generator
        self.generator = TrajectoryGenerator(
            N=N, L=L, T=T, dt=dt, n_snapshots=n_snapshots, device=device,
            filter_order=filter_order
        )
    
    def generate_initial_condition(self, init_type: str, seed: int, **kwargs) -> torch.Tensor:
        """Generate different types of initial conditions"""
        
        if init_type == 'random':
            # Random multiscale initial condition
            omega0_amplitude = kwargs.get('omega0_amplitude', 1.0)
            n_modes = kwargs.get('n_modes', 10)
            return self.generator.generate_random_initial_condition(
                omega0=omega0_amplitude, n_modes=n_modes, seed=seed
            )
        
        elif init_type == 'vortex':
            # Single vortex initial condition
            strength = kwargs.get('vortex_strength', 2.0)
            x_center = kwargs.get('x_center', np.pi)
            y_center = kwargs.get('y_center', np.pi)
            width = kwargs.get('vortex_width', 0.5)
            
            x = torch.linspace(0, self.L, self.N, device=self.device)
            y = torch.linspace(0, self.L, self.N, device=self.device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Gaussian vortex
            r_sq = (X - x_center)**2 + (Y - y_center)**2
            omega0 = strength * torch.exp(-r_sq / (2 * width**2))
            
            return omega0
        
        elif init_type == 'shear':
            # Shear layer initial condition from previous implementation
            amplitude = kwargs.get('shear_amplitude', 1.0)
            k_mode = kwargs.get('shear_k', 4)
            noise_level = kwargs.get('shear_noise', 0.1)
            
            y = torch.linspace(0, self.L, self.N, device=self.device)
            x = torch.linspace(0, self.L, self.N, device=self.device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            torch.manual_seed(seed)
            omega0 = amplitude * torch.tanh((Y - np.pi) / 0.2) * torch.sin(k_mode * X)
            omega0 += noise_level * torch.randn_like(omega0)
            return omega0

        elif init_type == 'shear_layer_instability':
            # 1. Shear Layer Instability (Kelvin-Helmholtz)
            delta = 0.15
            amp = 0.05
            
            x = torch.linspace(0, self.L, self.N, device=self.device)
            y = torch.linspace(0, self.L, self.N, device=self.device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # u = tanh((y-pi)/delta)
            # v = amp * sin(2x) * exp(-((y-pi)/delta)^2)
            # omega = dv/dx - du/dy
            
            # du/dy = sech^2((y-pi)/delta) * (1/delta)
            sech_sq = 1.0 / torch.cosh((Y - np.pi) / delta)**2
            du_dy = sech_sq / delta
            
            # dv/dx = amp * 2 * cos(2x) * exp(...)
            exp_term = torch.exp(-((Y - np.pi) / delta)**2)
            dv_dx = amp * 2 * torch.cos(2 * X) * exp_term
            
            omega0 = dv_dx - du_dy
            return omega0

        elif init_type == 'vortex_merger':
            # 2. Vortex Merger
            x1, y1 = np.pi - 0.8, np.pi
            x2, y2 = np.pi + 0.8, np.pi
            sigma = 0.3
            
            x = torch.linspace(0, self.L, self.N, device=self.device)
            y = torch.linspace(0, self.L, self.N, device=self.device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Vorticity defined directly
            omega0 = (torch.exp(-((X-x1)**2 + (Y-y1)**2)/(2*sigma**2)) - 
                      torch.exp(-((X-x2)**2 + (Y-y2)**2)/(2*sigma**2)))
            return omega0

        elif init_type == 'high_wavenumber':
            # 3. High Wavenumber Perturbation
            x = torch.linspace(0, self.L, self.N, device=self.device)
            y = torch.linspace(0, self.L, self.N, device=self.device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # u = 0.5*sin(2X+3Y) + 0.3*sin(5X-2Y) + 0.2*sin(8X+6Y)
            # v = 0.5*cos(3X-2Y) + 0.3*cos(4X+5Y) + 0.2*cos(7X-8Y)
            
            # du/dy terms
            # Term 1: 0.5*cos(2X+3Y)*3
            du_dy = (1.5 * torch.cos(2*X + 3*Y) + 
                     (-0.6) * torch.cos(5*X - 2*Y) + 
                     1.2 * torch.cos(8*X + 6*Y))
                     
            # dv/dx terms
            # Term 1: -0.5*sin(3X-2Y)*3
            dv_dx = (-1.5 * torch.sin(3*X - 2*Y) - 
                     1.2 * torch.sin(4*X + 5*Y) - 
                     1.4 * torch.sin(7*X - 8*Y))
            
            omega0 = dv_dx - du_dy
            return omega0

        elif init_type == 'taylor_green_noise':
            # 4. Taylor-Green with High-k Noise
            x = torch.linspace(0, self.L, self.N, device=self.device)
            y = torch.linspace(0, self.L, self.N, device=self.device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Base Taylor-Green: u = sin(X)cos(Y), v = -cos(X)sin(Y)
            # omega_TG = -2 sin(X) sin(Y)
            omega_tg = -2 * torch.sin(X) * torch.sin(Y)
            
            np.random.seed(42 + seed) # Consistent noise per seed
            noise_amp = 0.1
            
            omega_noise = torch.zeros_like(X)
            
            for k in range(8, 16):
                phase_x = np.random.rand() * 2 * np.pi
                phase_y = np.random.rand() * 2 * np.pi
                
                # u_noise += amp * sin(kX + px) -> du/dy = 0
                # v_noise += amp * sin(kY + py) -> dv/dx = 0
                # This simplistic noise in user snippet has curl = 0 ?
                # "u += ... sin(kX...)", "v += ... sin(kY...)"
                # du/dy = 0. dv/dx = 0. 
                # Wait, user snippet:
                # u += amp * sin(kX + phase_x)  => du/dy = 0
                # v += amp * sin(kY + phase_y)  => dv/dx = 0
                # So omega contribution is 0?
                # Ah, let's re-read carefully.
                # u depends on X. v depends on Y.
                # du/dy = 0. dv/dx = 0.
                # So vorticity is indeed 0 for this noise field if they only depend on longitudinal coord?
                # Wait, u is x-component. depends on X. 
                # v is y-component. depends on Y.
                # This is a compressive noise? div u = k cos(kX) + ... != 0.
                # If the flow is incompressible, this noise is invalid?
                # The user's snippet might be just illustrative or produces compressible flow projected to incompressible?
                # The Euler/NS solver projects to divergence-free. 
                # If I calculate omega for this noise, it is 0. 
                # So the noise has no rotational component?
                # Let's assume the user meant noise that AFFECTS vorticity.
                # OR, the projection step in the solver will kill the divergent part and keep the rotational part.
                # But if calculating omega from the formula gives 0, then after projection it is still 0 (since projection preserves vorticity in 2D or defines it).
                # Actually, in 2D streamfunction-vorticity, omega defines the flow.
                # If the user gives me u,v such that curl(u,v)=0, then omega=0.
                # Let's look at user snippet 4 again.
                # u += sin(kX), v += sin(kY).
                # curl = dv/dx - du/dy = 0 - 0 = 0.
                # This noise seems irrotational. It might do nothing in a vorticity-based solver unless I interpret it differently.
                # Maybe I should make the noise depend on both X and Y to have vorticity?
                # OR, I just faithfully implement it and result is just TG?
                
                # ALTERNATIVE: Maybe user meant u += sin(kY), v+= sin(kX)?
                # "u += noise_amp * np.sin(k*X + phase_x)"
                # "v += noise_amp * np.sin(k*Y + phase_y)"
                
                # Let's use a modification that guarantees vorticity.
                # High-k noise usually implies small vortices.
                # Let's add random Fourier modes to vorticity directly instead for this one?
                # Or rotate the user's noise?
                # I'll stick to user's idea but make it rotational: sin(kX + kY)?
                # Or just implement what user wrote and let them see it does nothing (or maybe I am wrong)?
                # If the user is an expert, maybe they expect the projection to do something?
                # But in 2D, a field with 0 vorticity IS potential flow. 
                # If boundary conditions are periodic, potential flow is just constant velocity.
                # The sines are periodic.
                # I'll add a comment and maybe "fix" it by making it depend on mixed coords?
                # "High-k noise" usually means energy in high k.
                # Let's implement the "High Wavenumber Perturbation" (ID 3) logic for noise?
                # Actually, let's look at the user message again.
                # "Viscosity differs... Viscosity damps high wavenumbers... Euler preserves them".
                # If the noise has 0 vorticity, there are no high wavenumbers in vorticity to damp.
                # I will slighty modify the noise to be rotational to be useful.
                # u += sin(kY), v += sin(kX). Then curl = k cos(kX) - k cos(kY) != 0.
                
                # Modified interpretation for useful noise:
                # u += sin(k*Y + phase)
                # v += sin(k*X + phase)
                
                du_dy_noise = noise_amp * k * torch.cos(k*Y + phase_y) 
                dv_dx_noise = noise_amp * k * torch.cos(k*X + phase_x)
                
                omega_noise += (dv_dx_noise - du_dy_noise)
                
            return omega_tg + omega_noise

        elif init_type == 'vortex_dipole':
            # 5. Concentrated Vortex Dipole
            d = 0.5
            sigma_dipole = 0.2
            
            x = torch.linspace(0, self.L, self.N, device=self.device)
            y = torch.linspace(0, self.L, self.N, device=self.device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Omega defined directly
            omega0 = (torch.exp(-((X - np.pi - d)**2 + (Y - np.pi)**2) / (2 * sigma_dipole**2)) - 
                      torch.exp(-((X - np.pi + d)**2 + (Y - np.pi)**2) / (2 * sigma_dipole**2)))
            omega0 *= 10.0 # Strong vorticity
            return omega0
            
        else:
            raise ValueError(f"Unknown initial condition type: {init_type}")


def generate_trajectory_batch(experiment: ViscosityExperiment, init_conditions: List[torch.Tensor], 
                             viscosities: List[float], solver_type: str, batch_size: int = None,
                             nu_h: float = 0.0, p: int = 4, adaptive: bool = False, cfl: float = 0.5) -> Dict:
    """Generate trajectories for a batch of initial conditions and viscosities using batch processing"""
    
    # Stack initial conditions into batch tensor
    omega0_batch = torch.stack(init_conditions)  # Shape: (n_initial, N, N)
    
    results = {
        'trajectories': [],
        'viscosities': [],
        'init_indices': [],
        'solver_type': solver_type
    }
    
    # Determine effective batch size
    n_initial = len(init_conditions)
    if batch_size is None:
        batch_size = n_initial  # Process all initial conditions at once
    
    for nu in viscosities:
        # Process initial conditions in batches
        for batch_start in range(0, n_initial, batch_size):
            batch_end = min(batch_start + batch_size, n_initial)
            batch_omega0 = omega0_batch[batch_start:batch_end]
            
            if solver_type == 'euler':
                # Euler solver - here we use nu_h if provided
                trajectory_batch = experiment.generator.euler_solver.solve(
                    batch_omega0, experiment.T, experiment.dt, nu=0.0, 
                    nu_h=nu_h, p=p, n_snapshots=experiment.n_snapshots,
                    adaptive=adaptive, cfl=cfl
                )
            elif solver_type == 'navier_stokes':
                # Navier-Stokes solver
                trajectory_batch = experiment.generator.ns_solver.solve(
                    batch_omega0, experiment.T, experiment.dt, nu=nu,
                    nu_h=nu_h, p=p, n_snapshots=experiment.n_snapshots,
                    adaptive=adaptive, cfl=cfl
                )
            else:
                raise ValueError(f"Unknown solver type: {solver_type}")
            
            # Unpack batch results
            for i, trajectory in enumerate(trajectory_batch.transpose(0, 1)):
                results['trajectories'].append(trajectory.cpu())
                results['viscosities'].append(nu)
                results['init_indices'].append(batch_start + i)
    
    return results


def save_experiment_data(output_path: Path, euler_results: Dict, ns_results: Dict,
                        viscosities: List[float], init_types: List[str], 
                        experiment_params: Dict):
    """Save experiment data to individual HDF5 files per initial condition folder"""
    
    n_initial = experiment_params['n_initial']
    n_viscosities = len(viscosities)
    
    # Organize data by initial condition
    euler_trajs = torch.stack(euler_results['trajectories'])
    ns_trajs = torch.stack(ns_results['trajectories'])
    
    # Correct reshape for ns_trajs is (n_viscosities, n_initial, n_snapshots, N, N)
    ns_by_visc_init = ns_trajs.reshape(n_viscosities, n_initial, *ns_trajs.shape[1:])
    
    for i in range(n_initial):
        init_dir = output_path / f"init_{i}"
        init_dir.mkdir(exist_ok=True)
        
        filename = init_dir / "trajectory.h5"
        with h5py.File(filename, 'w') as f:
            # Save metadata for this file
            meta_group = f.create_group('metadata')
            for key, value in experiment_params.items():
                if key == 'viscosities':
                    meta_group.create_dataset(key, data=np.array(value))
                elif isinstance(value, (list, tuple)):
                    meta_group.create_dataset(key, data=np.array(value, dtype='S'))
                else:
                    meta_group.attrs[key] = value
            
            meta_group.attrs['init_index'] = i
            meta_group.attrs['init_type'] = init_types[i % len(init_types)]
            
            # Euler trajectory
            euler_group = f.create_group('euler')
            euler_group.create_dataset('trajectory', data=euler_trajs[i].numpy())
            
            # Navier-Stokes trajectories
            ns_group = f.create_group('navier_stokes')
            for j, nu in enumerate(viscosities):
                nu_group = ns_group.create_group(f"visc_{nu:.0e}".replace('-', 'm'))
                nu_group.attrs['viscosity'] = nu
                nu_group.create_dataset('trajectory', data=ns_by_visc_init[j, i].numpy())
    
    print(f"✓ Data saved to per-init folders in {output_path}")


def create_comparison_plots(output_path: Path, euler_results: Dict, ns_results: Dict,
                           viscosities: List[float], init_types: List[str],
                           n_initial: int, experiment_params: Dict):
    """Create comprehensive comparison plots"""
    
    # 1. Initial condition vs final state comparison
    fig = plt.figure(figsize=(20, 12))
    
    # Organize data by initial condition
    euler_trajs = torch.stack(euler_results['trajectories'])
    ns_trajs = torch.stack(ns_results['trajectories'])
    
    n_viscosities = len(viscosities)
    
    # Reshape trajectories: [n_initial, n_viscosities, n_snapshots, N, N]
    # Note: trajectories are ordered by viscosity first, then initial condition
    euler_by_init = euler_trajs.reshape(n_initial, 1, *euler_trajs.shape[1:])
    ns_by_init = ns_trajs.reshape(n_viscosities, n_initial, *ns_trajs.shape[1:]).transpose(0, 1)
    
    # Plot grid: initial conditions vs viscosities
    n_cols = min(4, n_initial)
    n_rows = (n_viscosities + 1)  # +1 for Euler row
    
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    
    for col in range(min(n_cols, n_initial)):
        # Euler trajectory (top row)
        ax = fig.add_subplot(gs[0, col])
        euler_final = euler_by_init[col, 0, -1]  # Final state
        im = ax.imshow(euler_final, cmap='RdBu_r', aspect='equal')
        ax.set_title(f'Euler\nInit {col+1}')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Navier-Stokes trajectories (subsequent rows)
        for row, nu in enumerate(viscosities):
            ax = fig.add_subplot(gs[row+1, col])
            ns_final = ns_by_init[col, row, -1]  # Final state
            im = ax.imshow(ns_final, cmap='RdBu_r', aspect='equal')
            ax.set_title(f'NS ν={nu:.0e}\nInit {col+1}')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle('Final States: Euler vs Navier-Stokes Comparison\n' + 
                 f'T={experiment_params["T"]}, dt={experiment_params["dt"]}, ' +
                 f'Resolution={experiment_params["N"]}×{experiment_params["N"]}', 
                 fontsize=16)
    plt.tight_layout()
    summary_dir = output_path / "summary_plots"
    summary_dir.mkdir(exist_ok=True)
    plt.savefig(summary_dir / 'final_states_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Energy evolution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Calculate energies for first few initial conditions
    time_points = np.linspace(0, experiment_params['T'], experiment_params['n_snapshots'])
    
    for init_idx in range(min(4, n_initial)):
        ax = axes[init_idx]
        
        # Euler energy
        euler_traj = euler_by_init[init_idx, 0]  # [n_snapshots, N, N]
        euler_energy = torch.mean(euler_traj**2, dim=(1, 2)).numpy()
        ax.plot(time_points, euler_energy, 'k-', linewidth=2, label='Euler', alpha=0.8)
        
        # Navier-Stokes energies for different viscosities
        colors = plt.cm.viridis(np.linspace(0, 1, len(viscosities)))
        for nu_idx, (nu, color) in enumerate(zip(viscosities, colors)):
            ns_traj = ns_by_init[init_idx, nu_idx]  # [n_snapshots, N, N]
            ns_energy = torch.mean(ns_traj**2, dim=(1, 2)).numpy()
            ax.plot(time_points, ns_energy, color=color, linewidth=1.5, 
                   label=f'NS ν={nu:.0e}', alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        ax.set_title(f'Initial Condition {init_idx+1}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.suptitle('Energy Evolution Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path / "summary_plots" / 'energy_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Enstrophy evolution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for init_idx in range(min(4, n_initial)):
        ax = axes[init_idx]
        
        # Euler enstrophy
        euler_traj = euler_by_init[init_idx, 0]  # [n_snapshots, N, N]
        euler_enstrophy = torch.mean(euler_traj**2, dim=(1, 2)).numpy()
        ax.plot(time_points, euler_enstrophy, 'k-', linewidth=2, label='Euler', alpha=0.8)
        
        # NS enstrophy
        for nu_idx, (nu, color) in enumerate(zip(viscosities, colors)):
            ns_traj = ns_by_init[init_idx, nu_idx]
            ns_enstrophy = torch.mean(ns_traj**2, dim=(1, 2)).numpy()
            ax.plot(time_points, ns_enstrophy, color=color, linewidth=1.5,
                   label=f'NS ν={nu:.0e}', alpha=0.7)
            
        ax.set_xlabel('Time')
        ax.set_ylabel('Enstrophy')
        ax.set_title(f'Enstrophy - Init {init_idx+1}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.suptitle('Enstrophy Evolution Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path / "summary_plots" / 'enstrophy_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Vorticity Spectrum at Final Time
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    def get_spectrum1d(field_2d):
        # field_2d: (N, N)
        f_hat = torch.fft.fft2(field_2d)
        f_hat_shifted = torch.fft.fftshift(f_hat)
        
        N = field_2d.shape[0]
        k = torch.arange(-N//2, N//2)
        kx, ky = torch.meshgrid(k, k, indexing='ij')
        k_mag = torch.sqrt(kx**2 + ky**2)
        k_mag = k_mag.flatten().cpu().numpy()
        power = (torch.abs(f_hat_shifted)**2).flatten().cpu().numpy()
        
        # Bin by integer wavenumber
        k_bins = np.arange(0, N//2 + 1)
        spectrum = np.zeros(len(k_bins)-1)
        for i in range(len(k_bins)-1):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
            if np.any(mask):
                spectrum[i] = np.mean(power[mask])
        return k_bins[:-1], spectrum

    for init_idx in range(min(4, n_initial)):
        ax = axes[init_idx]
        
        # Euler Final Spectrum
        euler_final = euler_by_init[init_idx, 0, -1]
        k, spec_euler = get_spectrum1d(euler_final)
        ax.loglog(k, spec_euler, 'k-', linewidth=2, label='Euler', alpha=0.8)
        
        # NS Final Spectrum
        for nu_idx, (nu, color) in enumerate(zip(viscosities, colors)):
            ns_final = ns_by_init[init_idx, nu_idx, -1]
            k_ns, spec_ns = get_spectrum1d(ns_final)
            ax.loglog(k_ns, spec_ns, color=color, linewidth=1.5, label=f'NS ν={nu:.0e}', alpha=0.7)
            
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('Power Spectrum E(k)')
        ax.set_title(f'Vorticity Spectrum (t={experiment_params["T"]}) - Init {init_idx+1}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Final Vorticity Spectrum Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path / "summary_plots" / 'vorticity_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plots saved to {output_path}")


def create_per_init_plots(output_path: Path, euler_results: Dict, ns_results: Dict,
                           viscosities: List[float], n_initial: int, experiment_params: Dict):
    """Create comprehensive evolution and metric plots for each initial condition separately"""
    
    # Organize data
    euler_trajs = torch.stack(euler_results['trajectories'])
    ns_trajs = torch.stack(ns_results['trajectories'])
    n_viscosities = len(viscosities)
    n_snapshots = experiment_params['n_snapshots']
    ns_by_visc_init = ns_trajs.reshape(n_viscosities, n_initial, *ns_trajs.shape[1:])
    
    time_points = np.linspace(0, experiment_params['T'], n_snapshots)
    time_indices = [0, n_snapshots//4, n_snapshots//2, 3*n_snapshots//4, n_snapshots-1]
    time_labels = ['t=0', 't=T/4', 't=T/2', 't=3T/4', 't=T']
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_viscosities))

    for init_idx in range(n_initial):
        init_dir = output_path / f"init_{init_idx}"
        init_dir.mkdir(exist_ok=True)
        
        # 1. Evolution Snapshots Comparison
        fig = plt.figure(figsize=(20, 3 * (1 + n_viscosities)))
        gs = gridspec.GridSpec(1 + n_viscosities, 5, figure=fig, hspace=0.3, wspace=0.2)
        
        euler_data = euler_trajs[init_idx]
        vmin, vmax = float(euler_data.min()), float(euler_data.max())
        
        # Euler row
        for col, (t_idx, label) in enumerate(zip(time_indices, time_labels)):
            ax = fig.add_subplot(gs[0, col])
            im = ax.imshow(euler_data[t_idx].cpu(), cmap='RdBu_r', aspect='equal', vmin=vmin, vmax=vmax)
            ax.set_title(f"Euler\n{label}")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0: ax.set_ylabel("Euler", fontsize=12, fontweight='bold')
            
        # NS rows
        for row, nu in enumerate(viscosities):
            ns_data = ns_by_visc_init[row, init_idx]
            for col, (t_idx, label) in enumerate(zip(time_indices, time_labels)):
                ax = fig.add_subplot(gs[row+1, col])
                ax.imshow(ns_data[t_idx].cpu(), cmap='RdBu_r', aspect='equal', vmin=vmin, vmax=vmax)
                ax.set_title(f"NS ν={nu:.0e}\n{label}")
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0: ax.set_ylabel(f"ν={nu:.0e}", fontsize=12, fontweight='bold')
        
        plt.suptitle(f"Evolution Comparison - Init {init_idx+1}", fontsize=16)
        plt.savefig(init_dir / 'evolution.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Metrics (Energy, Enstrophy, Spectrum) for this init
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Energy
        euler_energy = torch.mean(euler_data**2, dim=(1, 2)).cpu().numpy()
        axes[0].plot(time_points, euler_energy, 'k-', linewidth=2, label='Euler')
        for j, (nu, color) in enumerate(zip(viscosities, colors)):
            ns_energy = torch.mean(ns_by_visc_init[j, init_idx]**2, dim=(1, 2)).cpu().numpy()
            axes[0].plot(time_points, ns_energy, color=color, label=f'ν={nu:.0e}')
        axes[0].set_title("Energy Evolution"); axes[0].set_yscale('log'); axes[0].legend()
        
        # Enstrophy
        euler_enstrophy = torch.mean(euler_data**2, dim=(1, 2)).cpu().numpy()
        axes[1].plot(time_points, euler_enstrophy, 'k-', linewidth=2, label='Euler')
        for j, (nu, color) in enumerate(zip(viscosities, colors)):
            ns_enstrophy = torch.mean(ns_by_visc_init[j, init_idx]**2, dim=(1, 2)).cpu().numpy()
            axes[1].plot(time_points, ns_enstrophy, color=color, label=f'ν={nu:.0e}')
        axes[1].set_title("Enstrophy Evolution"); axes[1].set_yscale('log'); axes[1].legend()
        
        # Spectrum at T
        def get_spec(field):
            f_hat = torch.fft.fft2(field)
            f_hat_shifted = torch.fft.fftshift(f_hat)
            N = field.shape[0]
            k = torch.arange(-N//2, N//2)
            kx, ky = torch.meshgrid(k, k, indexing='ij')
            k_mag = torch.sqrt(kx**2 + ky**2).flatten().cpu().numpy()
            power = (torch.abs(f_hat_shifted)**2).flatten().cpu().numpy()
            k_bins = np.arange(0, N//2 + 1)
            spec = np.zeros(len(k_bins)-1)
            for i in range(len(k_bins)-1):
                mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
                if np.any(mask): spec[i] = np.mean(power[mask])
            return k_bins[:-1], spec

        k, spec_e = get_spec(euler_data[-1])
        axes[2].loglog(k, spec_e, 'k-', linewidth=2, label='Euler')
        for j, (nu, color) in enumerate(zip(viscosities, colors)):
            _, spec_ns = get_spec(ns_by_visc_init[j, init_idx, -1])
            axes[2].loglog(k, spec_ns, color=color, label=f'ν={nu:.0e}')
        axes[2].set_title("Final Vorticity Spectrum"); axes[2].legend()
        
        plt.suptitle(f"Metrics Comparison - Init {init_idx+1}", fontsize=16)
        plt.tight_layout()
        plt.savefig(init_dir / 'metrics.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✓ Per-init plots saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Viscosity experiment: Euler vs Navier-Stokes comparison')
    parser.add_argument('--n-initial', type=int, default=20,
                       help='Number of different initial conditions (default: 20)')
    parser.add_argument('--viscosities', type=str, default='0.0,0.01,0.005,0.001',
                       help='Comma-separated viscosity values (default: 0.0,0.01,0.005,0.001)')
    parser.add_argument('--init-types', type=str, default='high_wavenumber',
                       help='Comma-separated initial condition types (default: high_wavenumber)')
    parser.add_argument('--resolution', type=int, default=256,
                       help='Grid resolution N (default: 256)')
    parser.add_argument('--time', type=float, default=10.0,
                       help='Simulation time (default: 10.0)')
    parser.add_argument('--dt', type=float, default=0.0005,
                       help='Initial time step (default: 0.0005)')
    parser.add_argument('--snapshots', type=int, default=20,
                       help='Number of snapshots per trajectory (default: 20)')
    parser.add_argument('--nu-h', type=float, default=0.0,
                       help='Hyperviscosity coefficient (default: 0.0)')
    parser.add_argument('--p-hyper', type=int, default=4,
                       help='Hyperviscosity power (default: 4)')
    parser.add_argument('--filter-order', type=int, default=16,
                       help='Exponential filter order (default: 16)')
    parser.add_argument('--adaptive', action='store_false', dest='fixed_dt',
                       help='Disable adaptive CFL-based time stepping')
    parser.set_defaults(fixed_dt=False)
    parser.add_argument('--cfl', type=float, default=0.2,
                       help='CFL safety factor for adaptive stepping (default: 0.2)')
    parser.add_argument('--output-dir', type=str, default='viscosity_experiment',
                       help='Base output directory (default: viscosity_experiment)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (default: auto)')
    parser.add_argument('--cores', type=int, default=None,
                       help='Number of CPU cores for parallel processing (default: auto)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for processing initial conditions (default: process all at once)')
    
    args = parser.parse_args()
    
    # Parse arguments
    viscosities = [float(v.strip()) for v in args.viscosities.split(',')]
    init_types = [t.strip() for t in args.init_types.split(',')]
    
    # Device selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_path = Path(args.output_dir)
    output_path = base_output_path / f"RUN_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create global summary plots directory
    (output_path / "summary_plots").mkdir(exist_ok=True)
    
    print("="*70)
    print("Viscosity Experiment: Euler vs Navier-Stokes Comparison")
    print("="*70)
    print(f"Initial conditions: {args.n_initial}")
    print(f"Initial condition types: {init_types}")
    print(f"Viscosities: {viscosities}")
    print(f"Grid resolution: {args.resolution}×{args.resolution}")
    print(f"Time span: 0 to {args.time}")
    print(f"Time step: {args.dt}")
    print(f"Snapshots: {args.snapshots}")
    print(f"Hyperviscosity: {'enabled (nu_h=' + str(args.nu_h) + ')' if args.nu_h > 0 else 'disabled'}")
    print(f"Spectral Stability: 2/3 Rule Truncation + Exp Filter (order {args.filter_order})")
    print(f"Adaptive time stepping: {not args.fixed_dt} (CFL: {args.cfl})")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size if args.batch_size else 'auto (all at once)'}")
    print(f"Output: {output_path.absolute()}")
    print("="*70)
    
    # Initialize experiment
    experiment = ViscosityExperiment(
        N=args.resolution,
        L=2*np.pi,
        T=args.time,
        dt=args.dt,
        n_snapshots=args.snapshots,
        device=device,
        filter_order=args.filter_order
    )
    
    total_start_time = time.time()
    
    # Generate initial conditions
    print("\n" + "="*50)
    print("Step 1: Generating initial conditions")
    print("="*50)
    
    initial_conditions = []
    for i in range(args.n_initial):
        # Cycle through init_types if we have more initial conditions than types
        init_type = init_types[i % len(init_types)]
        omega0 = experiment.generate_initial_condition(init_type, seed=i)
        initial_conditions.append(omega0)
        print(f"✓ Generated initial condition {i+1} (type: {init_type})")
    
    # Generate Euler trajectories
    print("\n" + "="*50)
    print("Step 2: Generating Euler trajectories")
    print("="*50)
    start_time = time.time()
    
    euler_results = generate_trajectory_batch(
        experiment, initial_conditions, [0.0], 'euler', batch_size=args.batch_size,
        nu_h=args.nu_h, p=args.p_hyper, adaptive=(not args.fixed_dt), cfl=args.cfl
    )
    
    euler_time = time.time() - start_time
    print(f"✓ Euler generation complete: {euler_time:.2f} seconds")
    
    # Generate Navier-Stokes trajectories
    print("\n" + "="*50)
    print("Step 3: Generating Navier-Stokes trajectories")
    print("="*50)
    start_time = time.time()
    
    ns_results = generate_trajectory_batch(
        experiment, initial_conditions, viscosities, 'navier_stokes', batch_size=args.batch_size,
        nu_h=args.nu_h, p=args.p_hyper, adaptive=(not args.fixed_dt), cfl=args.cfl
    )
    
    ns_time = time.time() - start_time
    print(f"✓ Navier-Stokes generation complete: {ns_time:.2f} seconds")
    
    # Save data
    print("\n" + "="*50)
    print("Step 4: Saving data")
    print("="*50)
    
    experiment_params = {
        'N': args.resolution,
        'L': 2*np.pi,
        'T': args.time,
        'dt': args.dt,
        'n_snapshots': args.snapshots,
        'n_initial': args.n_initial,
        'init_types': init_types,
        'viscosities': viscosities,
        'nu_h': args.nu_h,
        'p_hyper': args.p_hyper,
        'filter_order': args.filter_order
    }
    
    save_experiment_data(
        output_path, euler_results, ns_results, 
        viscosities, init_types, experiment_params
    )
    
    # Create plots
    print("\n" + "="*50)
    print("Step 5: Creating comparison plots")
    print("="*50)
    
    create_comparison_plots(
        output_path, euler_results, ns_results,
        viscosities, init_types, args.n_initial, experiment_params
    )
    
    # Create evolution snapshots for each initial condition
    create_per_init_plots(
        output_path, euler_results, ns_results,
        viscosities, args.n_initial, experiment_params
    )
    
    total_time = time.time() - total_start_time
    total_trajectories = len(euler_results['trajectories']) + len(ns_results['trajectories'])
    
    print(f"\n{'='*70}")
    print("✓ Viscosity experiment complete!")
    print(f"{'='*70}")
    print(f"Total trajectories generated: {total_trajectories}")
    print(f"  - Euler: {len(euler_results['trajectories'])}")
    print(f"  - Navier-Stokes: {len(ns_results['trajectories'])}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average time per trajectory: {total_time/total_trajectories:.3f} seconds")
    print(f"\nOutput directory: {output_path.absolute()}")
    print("Hierarchy created:")
    print("  - RUN_YYYYMMDD_HHMMSS/")
    print("    - summary_plots/ (Global comparison plots)")
    print("    - init_{i}/ (Data and plots for each initial condition)")
    print("      - trajectory.h5")
    print("      - evolution.png")
    print("      - metrics.png")


if __name__ == '__main__':
    main()