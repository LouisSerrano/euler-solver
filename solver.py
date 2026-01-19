"""
2D PDE Solvers for Euler, Diffusion, and Navier-Stokes equations
using pseudo-spectral methods with PyTorch.

Corrections Implemented:
1. Re-enabled 2/3 Rule De-aliasing on Advection Term (Fixes instability).
2. Added Exponential Spectral Filter (Fixes Gibbs phenomenon).
3. Added Hyperviscosity (order p=4) for Euler stability.
4. Added Adaptive Time-Stepping (CFL + Diffusion limits).
"""
import torch
import torch.fft as fft
import numpy as np
from typing import Tuple, Optional, Union

class SpectralSolver2D:
    """
    2D Solver using Real-FFTs for maximum speed and memory efficiency.
    Supports Batch dimensions (B, N, N).
    """

    def __init__(self, N: int, L: float = 2*np.pi, device: str = "cpu"):
        self.N = N
        self.L = L
        self.device = device

        # Physical Grid
        self.dx = L / N
        x = torch.linspace(0, L - self.dx, N, device=device)
        self.X, self.Y = torch.meshgrid(x, x, indexing="ij")

        # Wavenumbers for rfft2 
        # kx remains full size, ky is truncated to (N // 2 + 1)
        kx = 2 * np.pi * torch.fft.fftfreq(N, d=self.dx).to(device)
        ky = 2 * np.pi * torch.fft.rfftfreq(N, d=self.dx).to(device)
        self.KX, self.KY = torch.meshgrid(kx, ky, indexing="ij")

        self.k2 = self.KX**2 + self.KY**2
        self.k2_safe = self.k2.clone()
        self.k2_safe[0, 0] = 1.0

        # Pre-compute multipliers for gradients
        self.ikx = 1j * self.KX
        self.iky = 1j * self.KY

        # 2/3 dealiasing mask (Real-FFT shape)
        # This is CRITICAL for stability.
        k_max_global = (N // 2) * (2*np.pi/L)
        k_cut = (2/3) * k_max_global
        self.dealias = (torch.abs(self.KX) <= k_cut) & (torch.abs(self.KY) <= k_cut)

    def rfft2(self, f):
        return fft.rfft2(f)

    def irfft2(self, f_hat):
        return fft.irfft2(f_hat, s=(self.N, self.N))

class EulerSolver2D(SpectralSolver2D):
    def __init__(self, N: int, L: float = 2*np.pi, device: str = "cpu", filter_order: int = 16):
        super().__init__(N, L, device)
        
        # Exponential Filter for smoothing
        k_max = (N // 2) * (2 * np.pi / self.L)
        k_mag = torch.sqrt(self.k2)
        # alpha=36.0 ensures machine precision at k_max
        self.filter = torch.exp(-36.0 * (k_mag / k_max)**filter_order).to(device)

    def get_velocity(self, omega_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute velocity from vorticity (Fourier space)"""
        psi_hat = -omega_hat / self.k2_safe.to(omega_hat.dtype)
        psi_hat[..., 0, 0] = 0.0
        
        u = self.irfft2(-self.iky * psi_hat)
        v = self.irfft2(self.ikx * psi_hat)
        return u, v

    def rhs(self, omega: torch.Tensor, nu: float = 0.0, nu_h: float = 0.0, p: int = 4) -> torch.Tensor:
        """
        Compute RHS of the vorticity equation.
        Ensures stability via Exponential Filtering and Strict 2/3 Truncation.
        """
        # 1. Fourier Transform
        omega_hat = self.rfft2(omega)
        
        # 2. Strict Truncation & Spectral Filtering
        # We MUST zero out the top 1/3 frequencies BEFORE the product
        # to ensure the non-linear term (which doubles bandwidth) can be 
        # correctly de-aliased on the current grid.
        omega_hat = omega_hat * self.dealias * self.filter

        # 3. Compute Velocity & Gradients (from cleaned spectral state)
        u, v = self.get_velocity(omega_hat)
        omega_x = self.irfft2(self.ikx * omega_hat)
        omega_y = self.irfft2(self.iky * omega_hat)

        # 4. Compute Advection: -(u*wx + v*wy)
        adv = -(u * omega_x + v * omega_y)
        adv_hat = self.rfft2(adv)

        # 5. Apply 2/3 De-aliasing Truncation again
        # This removes the new high-frequencies created by the multiplication
        # that would otherwise wrap around (alias).
        rhs_hat = adv_hat * self.dealias
        
        # 6. Add Diffusion terms
        if nu > 0:
            rhs_hat -= nu * self.k2 * omega_hat

        # Hyperviscosity (numerical sink)
        if nu_h > 0:
            rhs_hat -= nu_h * (self.k2**p) * omega_hat

        return self.irfft2(rhs_hat)

    def step(self, omega: torch.Tensor, dt: float, nu: float = 0.0, nu_h: float = 0.0, p: int = 4) -> torch.Tensor:
        """RK4 Stepper"""
        k1 = self.rhs(omega, nu, nu_h, p)
        k2 = self.rhs(omega + 0.5 * dt * k1, nu, nu_h, p)
        k3 = self.rhs(omega + 0.5 * dt * k2, nu, nu_h, p)
        k4 = self.rhs(omega + dt * k3, nu, nu_h, p)
        return omega + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def get_stable_dt(self, omega: torch.Tensor, cfl: float, nu: float) -> float:
        """Compute max stable dt based on both CFL and Diffusion limits"""
        # 1. Advection Limit (CFL)
        omega_hat = self.rfft2(omega) * self.filter
        u, v = self.get_velocity(omega_hat)
        u_mag = torch.sqrt(u**2 + v**2)
        max_vel = torch.max(u_mag).item()
        
        if max_vel < 1e-8:
            dt_adv = 1.0
        else:
            dt_adv = cfl * self.dx / max_vel

        # 2. Diffusion Limit (Von Neumann stability for explicit diffusion)
        # dt < C * dx^2 / nu
        if nu > 1e-9:
            dt_diff = 0.5 * (self.dx**2) / nu
        else:
            dt_diff = 1e8 # Effectively infinite for Euler

        return min(dt_adv, dt_diff)

    def solve(self, omega0: torch.Tensor, T: float, dt: float, 
              nu: float = 0.0, nu_h: float = 0.0, p: int = 4,
              n_snapshots: int = 50, adaptive: bool = True, cfl: float = 0.5) -> torch.Tensor:
        
        omega = omega0.clone()
        B = omega0.shape[0]
        all_snapshots = torch.empty((n_snapshots, B, self.N, self.N), device=self.device)
        all_snapshots[0] = omega.clone()
        
        t = 0.0
        curr_snap = 1
        snapshot_times = torch.linspace(0, T, n_snapshots)
        
        print(f"Starting Solve: T={T}, N={self.N}, nu={nu}, nu_h={nu_h}, p={p}")

        if not adaptive:
            # Fixed time-step
            n_steps = int(T / dt)
            snapshot_interval = max(1, n_steps // (n_snapshots - 1))
            for step in range(n_steps):
                omega = self.step(omega, dt, nu, nu_h, p)
                if (step + 1) % snapshot_interval == 0 and curr_snap < n_snapshots:
                    all_snapshots[curr_snap] = omega.clone()
                    curr_snap += 1
        else:
            # Adaptive time-step
            while t < T and curr_snap < n_snapshots:
                # Calculate stable dt
                dt_stable = self.get_stable_dt(omega, cfl, nu)
                
                # Determine step to take (don't overstep next snapshot)
                time_to_snap = snapshot_times[curr_snap].item() - t
                dt_step = min(dt_stable, time_to_snap)
                
                # Minimum dt safety check
                if dt_step < 1e-7: 
                    dt_step = 1e-7

                omega = self.step(omega, dt_step, nu, nu_h, p)
                t += dt_step
                
                # Save snapshot if we reached the target time
                if abs(t - snapshot_times[curr_snap].item()) < 1e-6:
                    all_snapshots[curr_snap] = omega.clone()
                    curr_snap += 1
                    # Optional: Print progress
                    # print(f"  Snap {curr_snap}/{n_snapshots} | t={t:.3f} | dt={dt_step:.5f}")

        return all_snapshots

    # ---------------------------
    # Initial Conditions
    # ---------------------------
    def taylor_green_vortex(self, omega0: float = 1.0) -> torch.Tensor:
        return omega0 * (torch.sin(self.X) * torch.cos(self.Y) +
                         torch.cos(self.X) * torch.sin(self.Y))
    
    def random_vortices(self, n_vortices: int = 8, strength: float = 5.0) -> torch.Tensor:
        omega = torch.zeros_like(self.X)
        for _ in range(n_vortices):
            x0 = torch.rand(1, device=self.device) * self.L
            y0 = torch.rand(1, device=self.device) * self.L
            sign = torch.randint(0, 2, (1,), device=self.device) * 2 - 1
            vortex_strength = sign * strength * (0.5 + torch.rand(1, device=self.device))
            sigma = self.L / 20
            r_sq = (self.X - x0)**2 + (self.Y - y0)**2
            omega += vortex_strength * torch.exp(-r_sq / (2 * sigma**2))
        return omega

class DiffusionSolver2D(SpectralSolver2D):
    """Simple Diffusion Solver (unchanged)"""
    def rhs(self, omega: torch.Tensor, nu: float) -> torch.Tensor:
        return nu * self.irfft2(-self.k2 * self.rfft2(omega))

    def step(self, omega: torch.Tensor, dt: float, nu: float) -> torch.Tensor:
        k1 = self.rhs(omega, nu)
        k2 = self.rhs(omega + 0.5 * dt * k1, nu)
        k3 = self.rhs(omega + 0.5 * dt * k2, nu)
        k4 = self.rhs(omega + dt * k3, nu)
        return omega + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def solve(self, omega0: torch.Tensor, T: float, dt: float, nu: float, n_snapshots: int = 50):
        omega = omega0.clone()
        snapshots = [omega.clone()]
        n_steps = int(T / dt)
        interval = max(1, n_steps // (n_snapshots - 1))
        
        for i in range(n_steps):
            omega = self.step(omega, dt, nu)
            if (i+1) % interval == 0 and len(snapshots) < n_snapshots:
                snapshots.append(omega.clone())
        return torch.stack(snapshots)

class NavierStokesSolver2D(EulerSolver2D):
    """Wrapper for EulerSolver since logic is identical with nu > 0"""
    pass

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    N = 128
    L = 2 * np.pi
    T = 2.0
    
    solver = EulerSolver2D(N, L, device, filter_order=16)
    
    # 1. Setup Initial Condition
    omega0 = solver.random_vortices(n_vortices=12).unsqueeze(0)
    
    # 2. Run Euler (Inviscid)
    # Note: nu_h=1e-15 and p=4 is the "Sniper" setting for N=128/256
    print("\n--- Running Euler ---")
    euler_res = solver.solve(omega0, T, dt=0.0, nu=0.0, 
                             nu_h=1e-15, p=4, 
                             adaptive=True, cfl=0.5)
    
    # 3. Run Navier-Stokes (Viscous)
    print("\n--- Running Navier-Stokes (nu=1e-3) ---")
    ns_res = solver.solve(omega0, T, dt=0.0, nu=1e-3, 
                          nu_h=0.0,  # Usually don't need hypervisc if nu is high enough
                          adaptive=True, cfl=0.5)
    
    print("\nSuccess! Output shapes:", euler_res.shape, ns_res.shape)
    print(f"Euler Final Range: {euler_res[-1].min():.2f} to {euler_res[-1].max():.2f}")

if __name__ == "__main__":
    main()