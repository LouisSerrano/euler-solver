
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from compressible_solver import CompressibleSolver

def run_quadrants_experiment():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Setup Solver
    # Running at 200x200 resolution to resolve shocks clearly
    nx, ny = 200, 200
    solver = CompressibleSolver(nx=nx, ny=ny, lx=1.0, ly=1.0, bc_mode='replicate', device=device)
    
    # 2. Initial Conditions (Clawpack specifications)
    # Xs = 0.8, Ys = 0.8
    X, Y = solver.X, solver.Y
    
    # Masks for quadrants
    # Q1: x >= 0.8, y >= 0.8
    mask1 = (X >= 0.8) & (Y >= 0.8)
    # Q2: x < 0.8, y >= 0.8
    mask2 = (X < 0.8) & (Y >= 0.8)
    # Q3: x < 0.8, y < 0.8
    mask3 = (X < 0.8) & (Y < 0.8)
    # Q4: x >= 0.8, y < 0.8
    mask4 = (X >= 0.8) & (Y < 0.8)
    
    # Initialize primitive variables
    rho = torch.zeros_like(X)
    u = torch.zeros_like(X)
    v = torch.zeros_like(X)
    p = torch.zeros_like(X)
    
    # Quadrant 1
    # rpp=1.5, rpr=1.5, rpu=0, rpv=0
    rho[mask1] = 1.5
    p[mask1] = 1.5
    u[mask1] = 0.0
    v[mask1] = 0.0
    
    # Quadrant 2
    # rpp=0.3, rpr=0.5323, rpu=1.206, rpv=0
    rho[mask2] = 0.532258064516129
    p[mask2] = 0.3
    u[mask2] = 1.206045378311055
    v[mask2] = 0.0
    
    # Quadrant 3
    # rpp=0.029, rpr=0.138, rpu=1.206, rpv=1.206
    rho[mask3] = 0.137992831541219
    p[mask3] = 0.029032258064516
    u[mask3] = 1.206045378311055
    v[mask3] = 1.206045378311055
    
    # Quadrant 4
    # rpp=0.3, rpr=0.5323, rpu=0, rpv=1.206
    rho[mask4] = 0.532258064516129
    p[mask4] = 0.3
    u[mask4] = 0.0
    v[mask4] = 1.206045378311055
    
    # Add batch dimension
    rho = rho.unsqueeze(0)
    u = u.unsqueeze(0)
    v = v.unsqueeze(0)
    p = p.unsqueeze(0)
    
    # Create initial state Q0
    Q0 = solver.get_conservative(rho, u, v, p)
    
    # 3. Runs
    T_final = 0.8  # Longer simulation
    save_interval = 0.2
    
    # Function to run and collect snapshots
    def run_simulation(solver, Q_init, mu_val, label):
        print(f"Running {label} (mu={mu_val})...")
        Q = Q_init.clone()
        t = 0.0
        snapshots = []
        save_times = []
        next_save = save_interval
        
        # Save initial
        snapshots.append(Q.cpu())
        save_times.append(0.0)
        
        start_t = time.time()
        step_count = 0
        
        # CFL safety factor (Lowered for robustness with Viscous + Shocks + HLLC)
        cfl = 0.25 
        
        while t < T_final:
            # Adaptive DT
            dt = solver.compute_dt(Q, cfl=cfl, mu=mu_val)
            
            # Ensure we don't overstep next_save
            if t + dt > next_save:
                dt = next_save - t + 1e-9 # small epsilon to ensure we hit it
            
            Q = solver.step(Q, dt, mu=mu_val)
            t += dt
            step_count += 1
            
            # NaN Check
            if torch.isnan(Q).any():
                print(f"ERROR: NaN detected at t={t:.4f}, step={step_count}")
                break
            
            if t >= next_save - 1e-8: # tolerance
                print(f"  {label}: t={t:.2f} (steps={step_count})")
                snapshots.append(Q.cpu())
                save_times.append(t)
                next_save += save_interval
                
        print(f"{label} done in {time.time()-start_t:.2f}s (Total steps: {step_count})")
        return snapshots, save_times

    print("Starting simulations...")
    batch_Q0 = Q0.clone()
    euler_snaps, euler_times = run_simulation(solver, batch_Q0, 0.0, "Euler")
    
    mu_viscosity = 0.005 # Constant dynamic viscosity
    ns_snaps, ns_times = run_simulation(solver, batch_Q0, mu_viscosity, "Navier-Stokes")
    
    # 4. Visualization
    print("Generating plots...")
    
    # Plot 1: Time Evolution of Density
    print("  1. Time Evolution (quadrant_evolution.png)...")
    indices_to_plot = [1, 2, 3, 4] # t=0.2, 0.4, 0.6, 0.8
    fig_evo, ax_evo = plt.subplots(2, 4, figsize=(20, 10))
    
    # Global normalization for density
    all_rho = []
    valid_euler_snaps = [s for s in euler_snaps if not torch.isnan(s).any()]
    valid_ns_snaps = [s for s in ns_snaps if not torch.isnan(s).any()]
    
    for s in valid_euler_snaps + valid_ns_snaps:
        r, _, _, _ = solver.get_primitive(s)
        all_rho.append(r)
    
    if len(all_rho) > 0:
        all_rho = torch.cat(all_rho)
        vmin, vmax = all_rho.min().item(), all_rho.max().item()
    else:
        vmin, vmax = 0, 1
    
    for i, idx in enumerate(indices_to_plot):
        # Euler
        ax = ax_evo[0, i]
        if idx < len(euler_snaps):
            Q_e = euler_snaps[idx]
            rho_e, _, _, _ = solver.get_primitive(Q_e)
            im = ax.imshow(rho_e[0].cpu().T, origin='lower', extent=[0,1,0,1], cmap='jet', vmin=vmin, vmax=vmax)
            ax.set_title(f"Euler t={euler_times[idx]:.1f}")
        else:
            ax.set_title("Euler (Missing)")
        ax.axis('off')
        
        # NS
        ax = ax_evo[1, i]
        if idx < len(ns_snaps):
            Q_ns = ns_snaps[idx]
            rho_ns, _, _, _ = solver.get_primitive(Q_ns)
            im = ax.imshow(rho_ns[0].cpu().T, origin='lower', extent=[0,1,0,1], cmap='jet', vmin=vmin, vmax=vmax)
            ax.set_title(f"NS (mu={mu_viscosity}) t={ns_times[idx]:.1f}")
        else:
            ax.set_title("NS (Missing)")
        ax.axis('off')
        
    plt.tight_layout()
    fig_evo.subplots_adjust(right=0.92)
    cbar_ax = fig_evo.add_axes([0.93, 0.15, 0.02, 0.7])
    fig_evo.colorbar(im, cax=cbar_ax, label='Density')
    fig_evo.suptitle("Density Evolution: Euler vs Navier-Stokes", fontsize=16)
    fig_evo.savefig('quadrant_evolution.png', dpi=150)
    plt.close(fig_evo)

    # Plot 2: All Channels at Final Time
    print("  2. Final State All Channels (quadrant_all_channels.png)...")
    
    if len(euler_snaps) > 0 and len(ns_snaps) > 0:
        Q_e_final = euler_snaps[-1]
        rho_e, u_e, v_e, p_e = solver.get_primitive(Q_e_final)
        
        Q_ns_final = ns_snaps[-1]
        rho_ns, u_ns, v_ns, p_ns = solver.get_primitive(Q_ns_final)
        
        variables = [
            ('Density', rho_e, rho_ns, 'jet'),
            ('Pressure', p_e, p_ns, 'inferno'),
            ('Velocity X', u_e, u_ns, 'RdBu_r'),
            ('Velocity Y', v_e, v_ns, 'RdBu_r')
        ]
        
        fig_final, ax_final = plt.subplots(2, 4, figsize=(24, 12))
        
        for i, (name, var_e, var_ns, cmap) in enumerate(variables):
            vmin_c = min(var_e.min(), var_ns.min()).item()
            vmax_c = max(var_e.max(), var_ns.max()).item()
            
            # Euler
            ax = ax_final[0, i]
            im = ax.imshow(var_e[0].cpu().T, origin='lower', extent=[0,1,0,1], cmap=cmap, vmin=vmin_c, vmax=vmax_c)
            ax.set_title(f"Euler - {name}")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # NS
            ax = ax_final[1, i]
            im = ax.imshow(var_ns[0].cpu().T, origin='lower', extent=[0,1,0,1], cmap=cmap, vmin=vmin_c, vmax=vmax_c)
            ax.set_title(f"NS (mu={mu_viscosity}) - {name}")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        fig_final.suptitle(f"Euler vs Navier-Stokes (t={max(euler_times[-1], ns_times[-1]):.2f})", fontsize=20)
        plt.tight_layout()
        fig_final.savefig('quadrant_all_channels.png', dpi=150)
        plt.close(fig_final)
    else:
        print("Skipping All Channels plot due to missing data.")
    
    print("Done. Saved quadrant_evolution.png and quadrant_all_channels.png")

if __name__ == "__main__":
    run_quadrants_experiment()
