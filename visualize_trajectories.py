#!/usr/bin/env python3
"""
Visualization script to check trajectories from euler_diff_datasets/trajectories.h5
Displays sample trajectories for each class: Navier-Stokes, Euler, and Diffusion
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import torch
import os

def load_trajectories(filepath):
    """Load trajectories and labels from HDF5 file"""
    with h5py.File(filepath, 'r') as f:
        # Print dataset structure
        print("Dataset keys:", list(f.keys()))
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                print(f"{key}: shape={f[key].shape}, dtype={f[key].dtype}")
            else:
                print(f"{key}: Group with keys {list(f[key].keys())}")
        
        # Store trajectories separately to avoid memory issues
        trajectories_dict = {}
        
        # Load each class of data carefully to handle potential corruption
        for key, (label, name) in [('navier_stokes', (0, 'Navier-Stokes')), 
                                   ('euler', (1, 'Euler')), 
                                   ('diffusion', (2, 'Diffusion'))]:
            if key in f and isinstance(f[key], h5py.Dataset):
                try:
                    # Try to load data in chunks to handle potential corruption
                    dataset = f[key]
                    data_shape = dataset.shape
                    print(f"Loading {key} with shape {data_shape}...")
                    
                    # Load first few samples only to avoid memory/corruption issues
                    n_samples = min(5, data_shape[0]) if len(data_shape) > 0 else 0
                    if n_samples > 0:
                        trajectories_dict[key] = {
                            'data': dataset[:n_samples],
                            'label': label,
                            'name': name
                        }
                        print(f"Successfully loaded {n_samples} samples from {key}")
                    else:
                        print(f"Skipping {key} - empty dataset")
                        
                except Exception as e:
                    print(f"Error loading {key}: {e}")
                    print(f"Skipping {key} dataset due to corruption")
                    continue
        
        # Print dataset distribution
        print(f"\nDataset distribution:")
        for key, info in trajectories_dict.items():
            print(f"  {info['name']}: {len(info['data'])} samples")
    
    return trajectories_dict

def get_class_name(label):
    """Map numeric labels to class names"""
    class_map = {0: 'Navier-Stokes', 1: 'Euler', 2: 'Diffusion'}
    return class_map.get(label, f'Unknown-{label}')


def plot_single_trajectory(trajectory, title, save_path):
    """Plot a single trajectory with detailed snapshots"""
    
    trajectory_np = trajectory if isinstance(trajectory, np.ndarray) else trajectory.cpu().numpy()
    n_snapshots = len(trajectory_np)
    
    # Create figure with 5 rows x 8 columns = 40 snapshots
    fig, axes = plt.subplots(5, 8, figsize=(32, 20))
    axes = axes.flatten()
    
    # Use all available snapshots up to 40
    snapshots_to_show = min(n_snapshots, 40)
    
    for i in range(snapshots_to_show):
        ax = axes[i]
        
        # Find global vmin/vmax for consistent coloring
        vmax = max([np.abs(trajectory_np[j]).max() for j in range(n_snapshots)])
        vmin = -vmax
        
        im = ax.imshow(trajectory_np[i], cmap='RdBu_r', origin='lower', 
                      vmin=vmin, vmax=vmax)
        ax.set_title(f't = {i * 2.0 / (n_snapshots-1):.2f}', fontsize=8)
        ax.axis('off')
        # Only add colorbar to edge plots to reduce clutter
        if i % 8 == 0 or i % 8 == 7:  # First and last column
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(snapshots_to_show, 40):
        axes[i].axis('off')
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def create_comparison_grid(trajectories_dict, save_dir):
    """Create comparison grids for different classes"""
    
    class_names = list(trajectories_dict.keys())
    n_classes = len(class_names)
    
    if n_classes == 0:
        return
        
    # Create comparison figure
    fig, axes = plt.subplots(n_classes, 6, figsize=(24, 4*n_classes))
    
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    
    for i, class_name in enumerate(class_names):
        traj_data = trajectories_dict[class_name]
        trajectory = traj_data['data']
        
        # Use first trajectory for visualization
        if len(trajectory) > 0:
            first_traj = trajectory[0]  # First sample
            
            # Plot 6 time snapshots spread across the trajectory
            n_snapshots = len(first_traj)
            time_indices = np.linspace(0, n_snapshots-1, 6, dtype=int)
            
            # Find vmax for this trajectory
            vmax = np.abs(first_traj).max()
            vmin = -vmax
            
            for j, t_idx in enumerate(time_indices):
                im = axes[i, j].imshow(first_traj[t_idx], cmap='RdBu_r', 
                                     origin='lower', vmin=vmin, vmax=vmax)
                if i == 0:  # Only label top row
                    axes[i, j].set_title(f't = {t_idx / (n_snapshots-1):.2f}')
                axes[i, j].axis('off')
                
                # Add colorbar to first and last columns
                if j in [0, 5]:
                    plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
        
        # Add row label
        label = f"{traj_data['name']}\n({len(trajectory)} samples)"
        axes[i, 0].text(-0.15, 0.5, label, rotation=0, ha='right', va='center',
                      transform=axes[i, 0].transAxes, fontsize=12)
    
    plt.suptitle('Trajectory Class Comparison', fontsize=16)
    plt.tight_layout()
    
    save_path = f"{save_dir}/class_comparison.png"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")

def plot_static_trajectories(trajectories_dict, samples_per_class=2):
    """Plot static snapshots of trajectories for each class"""
    
    class_names = list(trajectories_dict.keys())
    n_classes = len(class_names)
    
    if n_classes == 0:
        print("No trajectory data found")
        return
    
    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(15, 4*n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    if samples_per_class == 1:
        axes = axes.reshape(-1, 1)
    
    for i, class_name in enumerate(class_names):
        traj_data = trajectories_dict[class_name]
        trajectories = traj_data['data']
        
        # Select random samples
        n_samples = min(samples_per_class, len(trajectories))
        selected_indices = np.random.choice(len(trajectories), n_samples, replace=False)
        
        for j, idx in enumerate(selected_indices):
            trajectory = trajectories[idx]
            
            # Show final timestep
            final_frame = trajectory[-1]
            
            # Handle different trajectory shapes
            if len(final_frame.shape) == 2:  # 2D field
                im = axes[i, j].imshow(final_frame, cmap='RdBu_r', origin='lower')
                plt.colorbar(im, ax=axes[i, j])
            elif len(final_frame.shape) == 3:  # Multi-component field
                # Show magnitude for vector fields
                if final_frame.shape[-1] == 2:  # 2-component vector field
                    magnitude = np.sqrt(final_frame[:, :, 0]**2 + final_frame[:, :, 1]**2)
                    im = axes[i, j].imshow(magnitude, cmap='RdBu_r', origin='lower')
                    plt.colorbar(im, ax=axes[i, j])
                else:  # Show first component
                    im = axes[i, j].imshow(final_frame[:, :, 0], cmap='RdBu_r', origin='lower')
                    plt.colorbar(im, ax=axes[i, j])
            
            axes[i, j].set_title(f'{traj_data["name"]} - Sample {j+1}')
            axes[i, j].set_xlabel('x')
            axes[i, j].set_ylabel('y')
    
    plt.tight_layout()
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/trajectory_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_time_evolution(trajectories_dict, class_name=None, sample_idx=0):
    """Plot time evolution of a single trajectory"""
    if class_name is not None:
        if class_name not in trajectories_dict:
            print(f"No samples found for class {class_name}")
            return
        traj_data = trajectories_dict[class_name]
        trajectories = traj_data['data']
        trajectory = trajectories[min(sample_idx, len(trajectories)-1)]
        label_name = traj_data['name']
    else:
        # Use first available class
        class_name = list(trajectories_dict.keys())[0]
        traj_data = trajectories_dict[class_name]
        trajectories = traj_data['data']
        trajectory = trajectories[sample_idx]
        label_name = traj_data['name']
    
    n_timesteps = trajectory.shape[0]
    timesteps_to_show = min(6, n_timesteps)
    step_size = max(1, n_timesteps // timesteps_to_show)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, t_idx in enumerate(range(0, min(timesteps_to_show * step_size, n_timesteps), step_size)):
        if i >= 6:
            break
        
        frame = trajectory[t_idx]
        
        if len(frame.shape) == 2:  # 2D scalar field
            im = axes[i].imshow(frame, cmap='RdBu_r', origin='lower')
            plt.colorbar(im, ax=axes[i])
        elif len(frame.shape) == 3:  # Multi-component field
            if frame.shape[-1] == 2:  # Vector field
                magnitude = np.sqrt(frame[:, :, 0]**2 + frame[:, :, 1]**2)
                im = axes[i].imshow(magnitude, cmap='RdBu_r', origin='lower')
                # Add velocity arrows (downsampled)
                skip = max(1, frame.shape[0] // 10)
                y, x = np.meshgrid(range(0, frame.shape[0], skip), range(0, frame.shape[1], skip), indexing='ij')
                u = frame[::skip, ::skip, 0]
                v = frame[::skip, ::skip, 1]
                axes[i].quiver(x, y, u, v, alpha=0.7, color='white', scale_units='xy')
                plt.colorbar(im, ax=axes[i])
            else:  # Show first component
                im = axes[i].imshow(frame[:, :, 0], cmap='RdBu_r', origin='lower')
                plt.colorbar(im, ax=axes[i])
        
        axes[i].set_title(f't = {t_idx}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
    
    fig.suptitle(f'{label_name} - Time Evolution (Sample {sample_idx})', fontsize=16)
    plt.tight_layout()
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig(f'visualizations/time_evolution_{class_name}.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_animation(trajectories_dict, class_name=None, sample_idx=0):
    """Create an animated visualization of trajectory evolution"""
    if class_name is not None:
        if class_name not in trajectories_dict:
            print(f"No samples found for class {class_name}")
            return
        traj_data = trajectories_dict[class_name]
        trajectories = traj_data['data']
        trajectory = trajectories[min(sample_idx, len(trajectories)-1)]
        label_name = traj_data['name']
    else:
        # Use first available class
        class_name = list(trajectories_dict.keys())[0]
        traj_data = trajectories_dict[class_name]
        trajectories = traj_data['data']
        trajectory = trajectories[sample_idx]
        label_name = traj_data['name']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Initialize plot
    frame = trajectory[0]
    if len(frame.shape) == 2:
        im = ax.imshow(frame, cmap='viridis', origin='lower', animated=True)
        plt.colorbar(im, ax=ax)
    elif len(frame.shape) == 3 and frame.shape[-1] == 2:
        magnitude = np.sqrt(frame[:, :, 0]**2 + frame[:, :, 1]**2)
        im = ax.imshow(magnitude, cmap='viridis', origin='lower', animated=True)
        plt.colorbar(im, ax=ax)
    else:
        im = ax.imshow(frame[:, :, 0], cmap='viridis', origin='lower', animated=True)
        plt.colorbar(im, ax=ax)
    
    ax.set_title(f'{label_name} - Animation (Sample {sample_idx})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    def animate(frame_idx):
        frame = trajectory[frame_idx]
        if len(frame.shape) == 2:
            im.set_array(frame)
        elif len(frame.shape) == 3 and frame.shape[-1] == 2:
            magnitude = np.sqrt(frame[:, :, 0]**2 + frame[:, :, 1]**2)
            im.set_array(magnitude)
        else:
            im.set_array(frame[:, :, 0])
        
        ax.set_title(f'{label_name} - t={frame_idx} (Sample {sample_idx})')
        return [im]
    
    # Create animation with every 5th frame for speed
    frames_to_animate = range(0, trajectory.shape[0], max(1, trajectory.shape[0] // 50))
    anim = FuncAnimation(fig, animate, frames=frames_to_animate, 
                        interval=100, blit=True, repeat=True)
    
    plt.show()
    return anim

def main():
    parser = argparse.ArgumentParser(description='Visualize trajectories from HDF5 dataset')
    parser.add_argument('--file', default="training_data/euler_trajectories.h5", #'euler_diff_datasets/trajectories.h5', 
                       help='Path to HDF5 trajectory file')
    parser.add_argument('--mode', choices=['static', 'evolution', 'animation'], 
                       default='static', help='Visualization mode')
    parser.add_argument('--class', type=str, dest='class_name',
                       choices=['navier_stokes', 'euler', 'diffusion'],
                       help='Specific class to visualize')
    parser.add_argument('--sample', type=int, default=0,
                       help='Sample index within class')
    parser.add_argument('--samples-per-class', type=int, default=2,
                       help='Number of samples per class for static mode')
    parser.add_argument('--detailed', action='store_true',
                       help='Create detailed individual trajectory plots')
    parser.add_argument('--comparison', action='store_true',
                       help='Create comparison grid plots')
    
    args = parser.parse_args()
    
    print(f"Loading trajectories from {args.file}...")
    trajectories_dict = load_trajectories(args.file)
    
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    if args.mode == 'static':
        print("Creating static trajectory comparison...")
        plot_static_trajectories(trajectories_dict, args.samples_per_class)
    
    elif args.mode == 'evolution':
        print("Creating time evolution plot...")
        plot_time_evolution(trajectories_dict, args.class_name, args.sample)
    
    elif args.mode == 'animation':
        print("Creating animation...")
        anim = create_animation(trajectories_dict, args.class_name, args.sample)
    
    # Create detailed plots if requested
    if args.detailed:
        print("Creating detailed individual trajectory plots...")
        for class_name, traj_data in trajectories_dict.items():
            trajectories = traj_data['data']
            n_samples = min(10, len(trajectories))  # Save up to 10 samples
            
            # Select random samples
            if n_samples > 0:
                selected_indices = np.random.choice(len(trajectories), n_samples, replace=False)
                for i, idx in enumerate(selected_indices):
                    trajectory = trajectories[idx]
                    title = f"{traj_data['name']} - Sample {i+1}"
                    save_path = f"visualizations/individual/{class_name}_sample_{i+1}_detailed.png"
                    plot_single_trajectory(trajectory, title, save_path)
    
    # Create comparison grid if requested
    if args.comparison:
        print("Creating comparison grid...")
        create_comparison_grid(trajectories_dict, 'visualizations')
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()