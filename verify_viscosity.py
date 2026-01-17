
import torch
import numpy as np
from compressible_solver import CompressibleSolver

def test_viscosity_pure_shear():
    """
    Test 1: Pure Shear Flow (Couette-like local check)
    u(y) = A * y, v(x) = 0.
    Should result in constant shear stress tau_xy = mu * A.
    div(tau) should be 0 because tau is constant.
    Dissipation should be proportional to (du/dy)^2.
    """
    print("Test 1: Pure Shear Flow Verification")
    nx, ny = 32, 32
    solver = CompressibleSolver(nx=nx, ny=ny, lx=1.0, ly=1.0, device='cpu')
    
    # Setup state
    # rho = 1
    # u = y (linear profile)
    # v = 0
    # p = 1
    # nu = 0.1
    
    y = solver.Y
    rho = torch.ones_like(y)
    u = y.clone()
    v = torch.zeros_like(y)
    p = torch.ones_like(y)
    
    Q = solver.get_conservative(rho.unsqueeze(0), u.unsqueeze(0), v.unsqueeze(0), p.unsqueeze(0))
    
    nu = 0.1
    # mu = rho * nu = 1 * 0.1 = 0.1
    # tau_xy = mu * (du/dy + dv/dx) = 0.1 * (1 + 0) = 0.1
    
    visc_flux = solver.viscous_fluxes_explicit(Q, nu)
    # visc_flux shape: [1, 4, Nx, Ny] -> [rho_flux, mom_x, mom_y, E_flux]
    # Ideally diff terms should be zero because linear profile -> constant stress -> zero divergence.
    # However, at boundaries of the "patch" used in finite difference it might be non-zero if periodic.
    # Let's check the center of the domain where it's linear.
    
    # Check stresses internal logic? 
    # We can't access interal vars easily, but we can check the output update.
    # div(tau) should be 0.
    
    mom_x_update = visc_flux[0, 1, 5:25, 5:25] # Crop boundaries
    
    print(f"  Max Momentum X update (should be ~0): {mom_x_update.abs().max().item():.2e}")
    if mom_x_update.abs().max().item() < 1e-5:
        print("  [PASS] Momentum conservation in constant shear")
    else:
        print("  [FAIL] Momentum update non-zero (unexpected for linear shear)")

def test_viscous_decay():
    """
    Test 2: Sine wave decay
    u(y) = u0 + eps * sin(2*pi*y)
    Should decay at rate exp(-nu * k^2 * t)
    """
    print("\nTest 2: Viscous Decay Verification")
    nx, ny = 64, 64
    solver = CompressibleSolver(nx=nx, ny=ny, lx=1.0, ly=1.0, device='cpu')
    
    y = solver.Y
    rho = torch.ones_like(y)
    u0 = 0.0
    epsilon = 0.1
    k = 2 * np.pi
    u = u0 + epsilon * torch.sin(k * y)
    v = torch.zeros_like(y)
    p = torch.ones_like(y)
    
    Q = solver.get_conservative(rho.unsqueeze(0), u.unsqueeze(0), v.unsqueeze(0), p.unsqueeze(0))
    nu = 0.01
    dt = 0.001
    
    # Expected decay rate for u velocity: du/dt = nu * d2u/dy2
    # u(t) = exp(-nu * k^2 * t) * u(0)
    # in one step: u_new = u_old * (1 - nu * k^2 * dt) approx
    
    # Run one step
    Q_next = solver.step(Q, dt, nu)
    _, u_next, _, _ = solver.get_primitive(Q_next)
    
    # Measure amplitude ratio at peaks (e.g. y=0.25 -> sin=1)
    #Center index approx
    idx = nx//4
    
    amp_old = u[idx, idx].item() # sin(pi/2)=1
    amp_new = u_next[0, idx, idx].item()
    
    ratio_sim = amp_new / amp_old
    ratio_theory = np.exp(-nu * k**2 * dt)
    
    print(f"  Amplitude Ratio (Sim): {ratio_sim:.6f}")
    print(f"  Amplitude Ratio (Thy): {ratio_theory:.6f}")
    
    err = abs(ratio_sim - ratio_theory)
    if err < 1e-4:
        print("  [PASS] Viscous decay rate matches theory")
    else:
        print(f"  [FAIL] Decay rate mismatch (Error: {err:.2e})")

if __name__ == "__main__":
    test_viscosity_pure_shear()
    test_viscous_decay()
