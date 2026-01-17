#!/usr/bin/env python3
"""
Generate and visualize trajectories for high-frequency multiscale turbulence cases
Focus on showing the actual evolution dynamics
"""

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver import EulerSolver2D, NavierStokesSolver2D
import numpy as np
import os


def create_trajectory_comparison():
    """Generate trajectories for key parameter combinations"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize solvers
    euler_solver = EulerSolver2D(N=128, device=device)
    ns_solver = NavierStokesSolver2D(N=128, device=device)
    
    # Key parameter combinations to compare
    cases = [
        # Low frequency baseline
        {'n_modes': 16, 'slope': -3.0, 'nu': 0.0, 'name': 'Low_Freq_Euler'},
        {'n_modes': 16, 'slope': -3.0, 'nu': 0.01, 'name': 'Low_Freq_Viscous'},
        
        # Medium frequency
        {'n_modes': 24, 'slope': -3.0, 'nu': 0.0, 'name': 'Med_Freq_Euler'},
        {'n_modes': 24, 'slope': -3.0, 'nu': 0.01, 'name': 'Med_Freq_Viscous'},
        
        # High frequency
        {'n_modes': 32, 'slope': -3.0, 'nu': 0.0, 'name': 'High_Freq_Euler'},
        {'n_modes': 32, 'slope': -3.0, 'nu': 0.01, 'name': 'High_Freq_Viscous'},
        {'n_modes': 40, 'slope': -3.0, 'nu': 0.0, 'name': 'VHigh_Freq_Euler'},
        {'n_modes': 40, 'slope': -3.0, 'nu': 0.01, 'name': 'VHigh_Freq_Viscous'},
        
        # Energy slope variations (high frequency)
        {'n_modes': 32, 'slope': -2.5, 'nu': 0.01, 'name': 'Shallow_Slope'},
        {'n_modes': 32, 'slope': -3.5, 'nu': 0.01, 'name': 'Steep_Slope'},
        
        # High viscosity cases
        {'n_modes': 32, 'slope': -3.0, 'nu': 0.05, 'name': 'High_Visc_32'},
        {'n_modes': 40, 'slope': -3.0, 'nu': 0.05, 'name': 'High_Visc_40'},
    ]
    
    trajectories = {}
    
    print(f"\nGenerating {len(cases)} trajectory cases...")
    
    for i, case in enumerate(cases):
        print(f"  [{i+1}/{len(cases)}] {case['name']}: n_modes={case['n_modes']}, slope={case['slope']}, ν={case['nu']}")
        
        # Choose solver
        solver = euler_solver if case['nu'] == 0.0 else ns_solver
        
        # Create initial condition
        omega0 = solver.multiscale_turbulence(n_modes=case['n_modes'], energy_slope=case['slope'])
        
        # Set simulation parameters
        T = 2.0  # Longer time for better evolution
        n_snapshots = 12  # More snapshots
        
        # Choose timestep
        if case['nu'] == 0.0:
            dt = 0.005
        elif case['nu'] <= 0.01:
            dt = 0.002
        else:
            dt = 0.001
        
        # Run simulation
        trajectory = solver.solve(omega0, T=T, dt=dt, nu=case['nu'], n_snapshots=n_snapshots)
        
        # Store results
        trajectories[case['name']] = {
            'trajectory': trajectory,
            'params': case,
            'initial_max': omega0.max().item(),
            'final_max': trajectory[-1].max().item(),
            'initial_energy': (omega0**2).mean().item(),
            'final_energy': (trajectory[-1]**2).mean().item()
        }
        
        print(f"    Initial max: {omega0.max().item():.3f}, Final max: {trajectory[-1].max().item():.3f}")
    
    return trajectories


def plot_single_trajectory(trajectory, title, save_path):
    """Plot a single trajectory with detailed snapshots"""
    
    trajectory_np = trajectory.cpu().numpy() if torch.is_tensor(trajectory) else trajectory
    n_snapshots = len(trajectory_np)
    
    # Create figure with 3 rows x 4 columns = 12 snapshots
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(min(n_snapshots, 12)):
        ax = axes[i]
        
        # Find global vmin/vmax for consistent coloring
        vmax = max([np.abs(trajectory_np[j]).max() for j in range(n_snapshots)])
        vmin = -vmax
        
        im = ax.imshow(trajectory_np[i], cmap='RdBu_r', origin='lower', 
                      vmin=vmin, vmax=vmax)
        ax.set_title(f't = {i * 2.0 / (n_snapshots-1):.2f}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(min(n_snapshots, 12), 12):
        axes[i].axis('off')
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def create_comparison_grid(trajectories, comparison_sets, save_dir):
    """Create comparison grids for different aspects"""
    
    for comp_name, case_names in comparison_sets.items():
        print(f"\nCreating {comp_name} comparison...")
        
        n_cases = len(case_names)
        fig, axes = plt.subplots(n_cases, 6, figsize=(24, 4*n_cases))
        
        if n_cases == 1:
            axes = axes.reshape(1, -1)
        
        for i, case_name in enumerate(case_names):
            if case_name not in trajectories:
                continue
                
            traj_data = trajectories[case_name]
            trajectory = traj_data['trajectory']
            trajectory_np = trajectory.cpu().numpy() if torch.is_tensor(trajectory) else trajectory
            
            # Plot 6 time snapshots
            time_indices = [0, 2, 4, 6, 8, 11]  # Spread across the trajectory
            
            # Find vmax for this trajectory
            vmax = np.abs(trajectory_np).max()
            vmin = -vmax
            
            for j, t_idx in enumerate(time_indices):
                if t_idx < len(trajectory_np):
                    im = axes[i, j].imshow(trajectory_np[t_idx], cmap='RdBu_r', 
                                         origin='lower', vmin=vmin, vmax=vmax)
                    if i == 0:  # Only label top row
                        axes[i, j].set_title(f't = {t_idx * 2.0 / 11:.2f}')
                    axes[i, j].axis('off')
                    
                    # Add colorbar to first and last columns
                    if j in [0, 5]:
                        plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
            
            # Add row label
            params = traj_data['params']
            label = f"{case_name}\nn={params['n_modes']}, s={params['slope']}, ν={params['nu']}"
            axes[i, 0].text(-0.15, 0.5, label, rotation=0, ha='right', va='center',
                          transform=axes[i, 0].transAxes, fontsize=10)
        
        plt.suptitle(f'{comp_name} Comparison', fontsize=16)
        plt.tight_layout()
        
        save_path = f"{save_dir}/{comp_name.lower().replace(' ', '_')}_comparison.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")


def main():
    print("="*70)
    print("High-Frequency Multiscale Turbulence Trajectory Visualization")
    print("="*70)
    
    # Generate all trajectories
    trajectories = create_trajectory_comparison()
    
    # Create output directory
    save_dir = 'visualizations/high_freq_trajectories'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save individual trajectory plots
    print(f"\nSaving individual trajectory plots...")
    for case_name, traj_data in trajectories.items():
        trajectory = traj_data['trajectory']
        params = traj_data['params']
        
        title = f"{case_name}: n_modes={params['n_modes']}, slope={params['slope']}, ν={params['nu']}"
        save_path = f"{save_dir}/individual/{case_name}_trajectory.png"
        
        plot_single_trajectory(trajectory, title, save_path)
    
    # Create comparison grids
    print(f"\nCreating comparison grids...")
    
    comparison_sets = {
        'Frequency_Progression_Euler': [
            'Low_Freq_Euler', 'Med_Freq_Euler', 'High_Freq_Euler', 'VHigh_Freq_Euler'
        ],
        'Frequency_Progression_Viscous': [
            'Low_Freq_Viscous', 'Med_Freq_Viscous', 'High_Freq_Viscous', 'VHigh_Freq_Viscous'
        ],
        'Euler_vs_Viscous_High_Freq': [
            'High_Freq_Euler', 'High_Freq_Viscous', 'VHigh_Freq_Euler', 'VHigh_Freq_Viscous'
        ],
        'Energy_Slope_Effects': [
            'Shallow_Slope', 'High_Freq_Viscous', 'Steep_Slope'
        ],
        'Viscosity_Effects_High_Freq': [
            'High_Freq_Euler', 'High_Freq_Viscous', 'High_Visc_32'
        ],
        'Very_High_Frequency': [
            'VHigh_Freq_Euler', 'VHigh_Freq_Viscous', 'High_Visc_40'
        ]
    }
    
    create_comparison_grid(trajectories, comparison_sets, save_dir)
    
    # Print summary
    print(f"\n" + "="*70)
    print("Trajectory Analysis Summary")
    print("="*70)
    
    for case_name, traj_data in trajectories.items():
        params = traj_data['params']
        initial_energy = traj_data['initial_energy']
        final_energy = traj_data['final_energy']
        energy_decay = (initial_energy - final_energy) / initial_energy
        
        print(f"{case_name:20s}: Energy decay = {energy_decay:6.3f}, "
              f"Max: {traj_data['initial_max']:6.3f} → {traj_data['final_max']:6.3f}")
    
    print(f"\n✓ All trajectories saved in {save_dir}/")
    print("  - individual/: Individual trajectory evolution plots")
    print("  - *_comparison.png: Side-by-side comparison grids")


if __name__ == '__main__':
    main()