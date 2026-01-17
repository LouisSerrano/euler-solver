
import torch
import numpy as np
import matplotlib.pyplot as plt
from compressible_solver import CompressibleSolver

def taylor_green_vortex_ic(solver, U0=1.0, rho0=1.0, p0=1.0):
    """
    Initial condition: Taylor-Green Vortex
    Analytical solution: exponential decay of kinetic energy.
    """
    x, y = solver.X, solver.Y
    
    # 2D TGV setup
    # u = U0 sin(2 pi x) cos(2 pi y)
    # v = -U0 cos(2 pi x) sin(2 pi y)
    u = U0 * torch.sin(2*np.pi*x) * torch.cos(2*np.pi*y)
    v = -U0 * torch.cos(2*np.pi*x) * torch.sin(2*np.pi*y)
    
    # Pressure to balance: p = p0 - rho0 U0^2 / 4 * (cos(4pix) + cos(4piy))
    p = p0 - (rho0 * U0**2 / 4) * (torch.cos(4*np.pi*x) + torch.cos(4*np.pi*y))
    rho = rho0 * torch.ones_like(x)
    
    return solver.get_conservative(rho.unsqueeze(0), u.unsqueeze(0), v.unsqueeze(0), p.unsqueeze(0))

def compute_kinetic_energy(solver, Q):
    rho, u, v, _ = solver.get_primitive(Q)
    ke = 0.5 * rho * (u**2 + v**2)
    return ke.sum().item() * solver.dx * solver.dy # Integral

def compute_total_mass(solver, Q):
    rho, _, _, _ = solver.get_primitive(Q)
    return rho.sum().item() * solver.dx * solver.dy

def maximize_window():
    mng = plt.get_current_fig_manager()
    try:
        mng.resize(*mng.window.maxsize())
    except:
        pass

def run_verification():
    print("=== Taylor-Green Vortex Verification ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nx, ny = 128, 128
    # TGV is periodic
    solver = CompressibleSolver(nx=nx, ny=ny, lx=1.0, ly=1.0, bc_mode='circular', device=device)
    
    U0 = 1.0
    p0 = 100.0 # High pressure to approximate incompressible limit
    rho0 = 1.0
    Q_init = taylor_green_vortex_ic(solver, U0, rho0, p0)
    
    # Parameters
    mu_val = 0.01
    dt = 0.0005
    t_final = 0.5
    
    print(f"Running TGV with mu={mu_val} on {device}...")
    
    Q = Q_init.clone()
    t = 0.0
    times = []
    ke_history = []
    mass_history = []
    
    mass_0 = compute_total_mass(solver, Q)
    ke_0 = compute_kinetic_energy(solver, Q)
    
    # Expected decay rate
    # k = 2 * nu * (2*pi*k_w)^2 ? 
    # For TGV: u ~ exp(-2 * nu * (2pi)^2 * t) considering wavenumber k=1 (2pi)
    # KE ~ u^2 ~ exp(-4 * nu * 4 pi^2 * t) = exp(-16 pi^2 nu t)
    # nu = mu / rho = 0.01 / 1 = 0.01
    # Decay rate lambda = 16 * pi^2 * 0.01 approx 1.579
    decay_rate = 16 * (np.pi**2) * (mu_val / rho0)
    print(f"Theoretical Energy Decay Rate: {decay_rate:.4f}")
    
    while t < t_final:
        # Step
        # dt = solver.compute_dt(Q, cfl=0.4, mu=mu_val) 
        # Use fixed small dt for accurate decay comparison or adaptive?
        # Let's use adaptive
        dt = solver.compute_dt(Q, cfl=0.5, mu=mu_val)
        if t + dt > t_final: dt = t_final - t
        
        Q = solver.step(Q, dt, mu=mu_val)
        t += dt
        
        # Record stats
        times.append(t)
        ke_history.append(compute_kinetic_energy(solver, Q))
        mass_history.append(compute_total_mass(solver, Q))
        
        Q = solver.step(Q, dt, mu=mu_val)
        t += dt
        
        # Record stats
        times.append(t)
        ke_history.append(compute_kinetic_energy(solver, Q))
        mass_history.append(compute_total_mass(solver, Q))
        
        if len(times) % 50 == 0:
            phi = solver.compute_viscous_dissipation(Q, mu_val)
            print(f"t={t:.3f}, Mass Err={(mass_history[-1]-mass_0)/mass_0:.2e}, KE={ke_history[-1]:.4f}, Phi={phi:.4f}")

    # Analysis
    times = np.array(times)
    ke_norm = np.array(ke_history) / ke_0
    mass_err = (np.array(mass_history) - mass_0) / mass_0
    
    # Theoretical KE
    ke_theory = np.exp(-decay_rate * times)
    
    print(f"Final Mass Error: {mass_err[-1]:.2e}")
    if abs(mass_err[-1]) < 1e-4:
        print("✅ Mass Conservation: PASS")
    else:
        print("❌ Mass Conservation: FAIL")
        
    print(f"Final KE Ratio: {ke_norm[-1]:.4f} (Theory: {ke_theory[-1]:.4f})")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(times, mass_err, label="Mass Error")
    plt.title("Mass Conservation")
    plt.xlabel("Time")
    plt.ylabel("Rel. Error")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(times, ke_norm, 'b-', label="Simulated KE")
    plt.plot(times, ke_theory, 'r--', label="Theoretical Decay")
    plt.title(f"Kinetic Energy Decay (mu={mu_val})")
    plt.xlabel("Time")
    plt.ylabel("E(t) / E(0)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('verification_tgv.png')
    print("Saved verification_tgv.png")

if __name__ == "__main__":
    run_verification()
