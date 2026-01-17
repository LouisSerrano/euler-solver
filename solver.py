"""
2D PDE Solvers for Euler, Diffusion, and Navier-Stokes equations
using pseudo-spectral methods with PyTorch
"""

import torch
import torch.fft as fft
import numpy as np
from typing import Optional, Tuple

import torch
import torch.fft as fft
import numpy as np
from typing import Tuple

import torch
import torch.fft as fft
import numpy as np
from typing import Tuple, Union

#torch._C._jit_set_profiling_executor(False)
#deatorch._C._jit_set_profiling_mode(False)

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
        k_cut = (2/3) * (N//2) * (2*np.pi/L)
        self.dealias = (torch.abs(self.KX) <= k_cut) & (torch.abs(self.KY) <= k_cut)

    def rfft2(self, f):
        return fft.rfft2(f)

    def irfft2(self, f_hat):
        # irfft2 automatically knows the output should be real
        return fft.irfft2(f_hat, s=(self.N, self.N))

class EulerSolver2D(SpectralSolver2D):
    def __init__(self, N: int, L: float = 2*np.pi, device: str = "cpu", filter_order: int = 16):
        super().__init__(N, L, device)
        
        # 1. Pre-compute Exponential Filter
        # Form: exp(-alpha * (k / k_max)^p)
        # alpha=36.0 is standard (sets machine precision at k_max)
        k_max = (N // 2) * (2 * np.pi / self.L)
        k_mag = torch.sqrt(self.k2)
        self.filter = torch.exp(-36.0 * (k_mag / k_max)**filter_order).to(device)

    def get_velocity(self, omega_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute velocity from vorticity (Fourier space)"""
        psi_hat = -omega_hat / self.k2_safe.to(omega_hat.dtype)
        psi_hat[..., 0, 0] = 0.0
        
        u = self.irfft2(-self.iky * psi_hat)
        v = self.irfft2(self.ikx * psi_hat)
        return u, v

    def rhs(self, omega: torch.Tensor, nu: float = 0.0, nu_h: float = 0.0, p: int = 2) -> torch.Tensor:
        """
        Compute RHS using Real-FFT operations with filtering and hyperviscosity.
        omega: (B, N, N)
        nu: standard viscosity
        nu_h: hyperviscosity coefficient
        p: hyperviscosity power (p=2 -> del^4, p=4 -> del^8)
        """
        omega_hat = self.rfft2(omega)

        # Apply the smoothing filter in Fourier space immediately
        omega_hat = omega_hat * self.filter

        # Velocity
        u, v = self.get_velocity(omega_hat)

        # Vorticity Gradients
        omega_x = self.irfft2(self.ikx * omega_hat)
        omega_y = self.irfft2(self.iky * omega_hat)

        # Advection in physical space
        adv = -(u * omega_x + v * omega_y)

        # Transform back to Fourier
        rhs_hat = self.rfft2(adv)

        # Standard Diffusion: -nu * k^2 * omega_hat
        if nu > 0:
            rhs_hat = rhs_hat - nu * self.k2 * omega_hat

        # Hyperviscosity: -nu_h * (k^2)^p * omega_hat
        if nu_h > 0:
            rhs_hat = rhs_hat - nu_h * (self.k2**p) * omega_hat

        return self.irfft2(rhs_hat)

    def step(self, omega: torch.Tensor, dt: float, nu: float = 0.0, nu_h: float = 0.0, p: int = 2) -> torch.Tensor:
        """RK4 Stepper"""
        k1 = self.rhs(omega, nu, nu_h, p)
        k2 = self.rhs(omega + 0.5 * dt * k1, nu, nu_h, p)
        k3 = self.rhs(omega + 0.5 * dt * k2, nu, nu_h, p)
        k4 = self.rhs(omega + dt * k3, nu, nu_h, p)
        return omega + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def get_cfl_dt(self, omega: torch.Tensor, cfl: float = 0.5) -> float:
        """Compute max stable dt based on CFL condition"""
        omega_hat = self.rfft2(omega)
        u, v = self.get_velocity(omega_hat)
        max_vel = torch.max(torch.sqrt(u**2 + v**2)).item()
        if max_vel < 1e-8:
            return 1.0 # arbitrary large
        return cfl * self.dx / max_vel

    def solve(self, omega0: torch.Tensor, T: float, dt: float, 
              nu: float = 0.0, nu_h: float = 0.0, p: int = 2,
              n_snapshots: int = 50, adaptive: bool = False, cfl: float = 0.5) -> torch.Tensor:
        
        omega = omega0.clone()
        B = omega0.shape[0]
        all_snapshots = torch.empty((n_snapshots, B, self.N, self.N), device=self.device)
        all_snapshots[0] = omega.clone()
        
        t = 0.0
        curr_snap = 1
        snapshot_times = torch.linspace(0, T, n_snapshots)
        
        if not adaptive:
            n_steps = int(T / dt)
            snapshot_interval = max(1, n_steps // (n_snapshots - 1))
            for step in range(n_steps):
                omega = self.step(omega, dt, nu, nu_h, p)
                if (step + 1) % snapshot_interval == 0 and curr_snap < n_snapshots:
                    all_snapshots[curr_snap] = omega.clone()
                    curr_snap += 1
        else:
            # Adaptive time stepping
            while t < T and curr_snap < n_snapshots:
                dt_cfl = self.get_cfl_dt(omega, cfl)
                dt_actual = min(dt_cfl, snapshot_times[curr_snap] - t)
                
                omega = self.step(omega, dt_actual, nu, nu_h, p)
                t += dt_actual
                
                if t >= snapshot_times[curr_snap] - 1e-10:
                    all_snapshots[curr_snap] = omega.clone()
                    curr_snap += 1

        return all_snapshots

    # ---------------------------
    # Utility
    # ---------------------------
    def velocity(self, omega: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get velocity field from vorticity"""
        omega_hat = self.fft2(omega)
        psi_hat = -omega_hat / self.k2_safe
        psi_hat[0, 0] = 0.0
        u = self.ifft2(-self.iky * psi_hat)
        v = self.ifft2( self.ikx * psi_hat)
        return u, v

    def taylor_green_vortex(self, omega0: float = 1.0) -> torch.Tensor:
        """Initialize Taylor-Green vortex"""
        omega = omega0 * (torch.sin(self.X) * torch.cos(self.Y) +
                          torch.cos(self.X) * torch.sin(self.Y))
        return omega
    
    def random_vortices(self, n_vortices: int = 8, strength: float = 5.0) -> torch.Tensor:
        """Initialize random vortices"""
        omega = torch.zeros_like(self.X)
        
        for _ in range(n_vortices):
            # Random position
            x0 = torch.rand(1, device=self.device) * self.L
            y0 = torch.rand(1, device=self.device) * self.L
            
            # Random strength (positive or negative)
            sign = torch.randint(0, 2, (1,), device=self.device) * 2 - 1
            vortex_strength = sign * strength * (0.5 + torch.rand(1, device=self.device))
            
            # Gaussian vortex
            sigma = self.L / 20
            r_sq = (self.X - x0)**2 + (self.Y - y0)**2
            omega += vortex_strength * torch.exp(-r_sq / (2 * sigma**2))
        
        return omega
    
    def shear_layer(self, delta: float = 0.1, U: float = 1.0) -> torch.Tensor:
        """Initialize mixing layer with Kelvin-Helmholtz instability"""
        y_center = self.L / 2
        
        # Base velocity profile (hyperbolic tangent)
        tanh_profile = torch.tanh((self.Y - y_center) / delta)
        
        # Add small perturbation to trigger instability
        perturbation = 0.1 * U * torch.sin(2 * np.pi * self.X / self.L) * \
                      torch.exp(-((self.Y - y_center) / (2*delta))**2)
        
        # Convert to vorticity
        omega = -U / (delta * torch.cosh((self.Y - y_center) / delta)**2)
        omega += perturbation / delta
        
        return omega
    
    def double_shear(self, delta: float = 0.1, U: float = 1.0) -> torch.Tensor:
        """Initialize double shear layer"""
        y1 = self.L / 3
        y2 = 2 * self.L / 3
        
        # Two shear layers
        omega1 = -U / (delta * torch.cosh((self.Y - y1) / delta)**2)
        omega2 = U / (delta * torch.cosh((self.Y - y2) / delta)**2)
        
        # Add perturbations
        pert1 = 0.1 * U * torch.sin(4 * np.pi * self.X / self.L) * \
                torch.exp(-((self.Y - y1) / (2*delta))**2)
        pert2 = 0.1 * U * torch.sin(6 * np.pi * self.X / self.L) * \
                torch.exp(-((self.Y - y2) / (2*delta))**2)
        
        return omega1 + omega2 + (pert1 + pert2) / delta

    def multiscale_turbulence(self, n_modes: int = 16, energy_slope: float = -3.0) -> torch.Tensor:
        """Initialize turbulent-like field with multiple scales"""
        omega = torch.zeros_like(self.X)
        
        for kx in range(1, n_modes + 1):
            for ky in range(1, n_modes + 1):
                k_mag = torch.sqrt(torch.tensor(kx**2 + ky**2, dtype=torch.float32))
                
                # Energy spectrum E(k) ∝ k^energy_slope
                amplitude = k_mag**energy_slope
                
                # Random phases
                phase_x = torch.rand(1, device=self.device) * 2 * np.pi
                phase_y = torch.rand(1, device=self.device) * 2 * np.pi
                
                # Add mode
                omega += amplitude * torch.sin(kx * 2*np.pi * self.X / self.L + phase_x) * \
                         torch.cos(ky * 2*np.pi * self.Y / self.L + phase_y)
        
        return omega

    def vortex_clusters(self, n_clusters: int = 4, vortices_per_cluster: int = 6) -> torch.Tensor:
        """Initialize clusters of vortices"""
        omega = torch.zeros_like(self.X)
        
        for cluster in range(n_clusters):
            # Cluster center
            cx = torch.rand(1, device=self.device) * self.L
            cy = torch.rand(1, device=self.device) * self.L
            cluster_radius = self.L / 8
            
            for _ in range(vortices_per_cluster):
                # Position within cluster
                angle = torch.rand(1, device=self.device) * 2 * np.pi
                radius = torch.rand(1, device=self.device) * cluster_radius
                x0 = cx + radius * torch.cos(angle)
                y0 = cy + radius * torch.sin(angle)
                
                # Periodic boundary handling
                x0 = x0 % self.L
                y0 = y0 % self.L
                
                # Vortex strength (alternating signs for dipole-like structures)
                strength = 5.0 * (1.0 - 2.0 * (torch.rand(1, device=self.device) > 0.5))
                
                # Vortex size
                sigma = self.L / 40
                r_sq = (self.X - x0)**2 + (self.Y - y0)**2
                omega += strength * torch.exp(-r_sq / (2 * sigma**2))
        
        return omega

    def dense_shear_lattice(self, n_layers: int = 6, base_strength: float = 2.0) -> torch.Tensor:
        """Initialize dense lattice of shear layers"""
        omega = torch.zeros_like(self.X)
        delta = self.L / (20 * n_layers)
        
        for i in range(n_layers):
            # Horizontal shear layers
            y_pos = (i + 0.5) * self.L / n_layers
            strength = base_strength * (1 + 0.5 * torch.sin(torch.tensor(2*np.pi*i/n_layers)))
            omega += strength / (delta * torch.cosh((self.Y - y_pos) / delta)**2)
            
            # Vertical shear layers (rotated 90 degrees)
            x_pos = (i + 0.5) * self.L / n_layers
            omega += strength / (delta * torch.cosh((self.X - x_pos) / delta)**2)
            
            # Add small-scale perturbations
            pert = 0.2 * strength * torch.sin((i+3) * 2*np.pi * self.X / self.L) * \
                   torch.cos((i+2) * 2*np.pi * self.Y / self.L)
            omega += pert / delta
        
        return omega

    def fourier_random_field(self, n_modes: int = 20, amplitude: float = 3.0) -> torch.Tensor:
        """Generate random field using Fourier coefficients"""
        omega_hat = torch.zeros((self.N, self.N), dtype=torch.complex64, device=self.device)
        
        for kx in range(-n_modes, n_modes + 1):
            for ky in range(-n_modes, n_modes + 1):
                if kx == 0 and ky == 0:
                    continue
                    
                # Map to array indices
                kx_idx = kx % self.N
                ky_idx = ky % self.N
                
                # Energy decay with wavenumber
                k_mag = torch.sqrt(torch.tensor(kx**2 + ky**2, dtype=torch.float32))
                energy = amplitude / (1 + k_mag**2)
                
                # Random complex coefficient
                real_part = torch.randn(1, device=self.device) * energy
                imag_part = torch.randn(1, device=self.device) * energy
                omega_hat[kx_idx, ky_idx] = real_part + 1j * imag_part
        
        # Ensure reality condition for real output
        omega_hat[0, 0] = 0  # Zero mean
        for kx in range(self.N):
            for ky in range(self.N):
                kx_conj = (-kx) % self.N
                ky_conj = (-ky) % self.N
                if kx != kx_conj or ky != ky_conj:
                    omega_hat[kx_conj, ky_conj] = torch.conj(omega_hat[kx, ky])
        
        return self.ifft2(omega_hat)

    def vorticity_to_velocity(self, omega: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert vorticity to velocity via streamfunction"""
        # Solve ∇²ψ = ω for streamfunction ψ
        psi = self.poisson_solve(omega)

        # Compute velocity: u = -∂ψ/∂y, v = ∂ψ/∂x
        psi_hat = self.fft2(psi)
        u = self.ifft2(-1j * self.ky * psi_hat)
        v = self.ifft2(1j * self.kx * psi_hat)

        return u, v

    def rhs_old(self, omega: torch.Tensor, nu: float = 0.0) -> torch.Tensor:
        """
        Compute RHS of vorticity equation:
        ∂ω/∂t = -u·∇ω + ν∇²ω
        """
        # Get velocity from vorticity
        u, v = self.vorticity_to_velocity(omega)

        # Compute vorticity gradient
        omega_x, omega_y = self.gradient(omega)

        # Advection term: -u·∇ω
        advection = -(u * omega_x + v * omega_y)

        # Dealias advection
        advection_hat = self.dealias_field(self.fft2(advection))
        advection = self.ifft2(advection_hat)

        # Diffusion term: ν∇²ω
        if nu > 0:
            diffusion = nu * self.laplacian(omega)
            return advection + diffusion

        return advection

    def rk4_step(self, omega: torch.Tensor, dt: float, nu: float = 0.0) -> torch.Tensor:
        """Fourth-order Runge-Kutta time step"""
        k1 = self.rhs(omega, nu)
        k2 = self.rhs(omega + 0.5 * dt * k1, nu)
        k3 = self.rhs(omega + 0.5 * dt * k2, nu)
        k4 = self.rhs(omega + dt * k3, nu)

        return omega + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def solve_old(self, omega0: torch.Tensor, T: float, dt: float,
              nu: float = 0.0, n_snapshots: int = 50) -> torch.Tensor:
        """
        Solve the Euler/Navier-Stokes equations

        Args:
            omega0: Initial vorticity field
            T: Total simulation time
            dt: Time step
            nu: Viscosity (0 for Euler)
            n_snapshots: Number of snapshots to save

        Returns:
            Tensor of shape (n_snapshots, N, N) containing vorticity snapshots
        """
        omega = omega0.clone()
        n_steps = int(T / dt)
        snapshot_interval = max(1, n_steps // (n_snapshots - 1))

        snapshots = []
        snapshots.append(omega.clone())

        for step in range(n_steps):
            omega = self.rk4_step(omega, dt, nu)

            if (step + 1) % snapshot_interval == 0 and len(snapshots) < n_snapshots:
                snapshots.append(omega.clone())

        # Ensure we have exactly n_snapshots
        while len(snapshots) < n_snapshots:
            snapshots.append(omega.clone())

        return torch.stack(snapshots[:n_snapshots])


class DiffusionSolver2D(SpectralSolver2D):
    """2D Diffusion equation solver"""

    def __init__(self, N: int, L: float = 2 * np.pi, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(N, L, device)

    def laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian using spectral methods"""
        f_hat = self.rfft2(f)
        laplacian_hat = -self.k2 * f_hat
        return self.irfft2(laplacian_hat)

    def rhs(self, omega: torch.Tensor, nu: float) -> torch.Tensor:
        """
        Compute RHS of diffusion equation:
        ∂ω/∂t = ν∇²ω
        """
        return nu * self.laplacian(omega)

    def rk4_step(self, omega: torch.Tensor, dt: float, nu: float) -> torch.Tensor:
        """Fourth-order Runge-Kutta time step"""
        k1 = self.rhs(omega, nu)
        k2 = self.rhs(omega + 0.5 * dt * k1, nu)
        k3 = self.rhs(omega + 0.5 * dt * k2, nu)
        k4 = self.rhs(omega + dt * k3, nu)

        return omega + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def solve(self, omega0: torch.Tensor, T: float, dt: float,
              nu: float, n_snapshots: int = 50) -> torch.Tensor:
        """
        Solve the diffusion equation

        Args:
            omega0: Initial field
            T: Total simulation time
            dt: Time step
            nu: Diffusion coefficient
            n_snapshots: Number of snapshots to save

        Returns:
            Tensor of shape (n_snapshots, N, N) containing field snapshots
        """
        omega = omega0.clone()
        n_steps = int(T / dt)
        snapshot_interval = max(1, n_steps // (n_snapshots - 1))

        snapshots = []
        snapshots.append(omega.clone())

        for step in range(n_steps):
            omega = self.rk4_step(omega, dt, nu)

            if (step + 1) % snapshot_interval == 0 and len(snapshots) < n_snapshots:
                snapshots.append(omega.clone())

        # Ensure we have exactly n_snapshots
        while len(snapshots) < n_snapshots:
            snapshots.append(omega.clone())

        return torch.stack(snapshots[:n_snapshots])

    def taylor_green_vortex(self, omega0: float = 1.0) -> torch.Tensor:
        """Initialize Taylor-Green vortex"""
        omega = omega0 * (torch.sin(self.X) * torch.cos(self.Y) +
                          torch.cos(self.X) * torch.sin(self.Y))
        return omega


class NavierStokesSolver2D(EulerSolver2D):
    """2D Navier-Stokes solver (same as Euler but with viscosity)"""

    def __init__(self, N: int, L: float = 2 * np.pi, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(N, L, device)

    def solve(self, omega0: torch.Tensor, T: float, dt: float,
              nu: float, nu_h: float = 0.0, p: int = 2, n_snapshots: int = 50, 
              adaptive: bool = False, cfl: float = 0.5) -> torch.Tensor:
        """
        Solve the Navier-Stokes equations (with viscosity)

        Args:
            omega0: Initial vorticity field
            T: Total simulation time
            dt: Time step
            nu: Viscosity
            nu_h: Hyperviscosity
            p: Hyperviscosity power
            n_snapshots: Number of snapshots to save
            adaptive: Use adaptive time stepping
            cfl: CFL safety factor

        Returns:
            Tensor of shape (n_snapshots, N, N) containing vorticity snapshots
        """
        return super().solve(omega0, T, dt, nu=nu, nu_h=nu_h, p=p, 
                             n_snapshots=n_snapshots, adaptive=adaptive, cfl=cfl)


def main():
    """Test the solvers with simple examples"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test parameters
    N = 64
    L = 2 * np.pi
    T = 1.0
    dt = 0.01
    
    # Test Euler solver
    print("\nTesting Euler solver...")
    euler_solver = EulerSolver2D(N, L, device)
    omega0 = euler_solver.taylor_green_vortex().unsqueeze(0)  # Add batch dimension
    
    euler_result = euler_solver.solve(omega0, T, dt, nu=0.0, n_snapshots=10)
    print(f"Euler simulation completed. Output shape: {euler_result.shape}")
    print(f"Initial vorticity range: [{omega0.min():.3f}, {omega0.max():.3f}]")
    print(f"Final vorticity range: [{euler_result[-1].min():.3f}, {euler_result[-1].max():.3f}]")
    
    # Test Navier-Stokes solver
    print("\nTesting Navier-Stokes solver...")
    ns_solver = NavierStokesSolver2D(N, L, device)
    omega0_ns = ns_solver.taylor_green_vortex().unsqueeze(0)
    
    ns_result = ns_solver.solve(omega0_ns, T, dt, nu=0.01, n_snapshots=10)
    print(f"Navier-Stokes simulation completed. Output shape: {ns_result.shape}")
    print(f"Initial vorticity range: [{omega0_ns.min():.3f}, {omega0_ns.max():.3f}]")
    print(f"Final vorticity range: [{ns_result[-1].min():.3f}, {ns_result[-1].max():.3f}]")
    
    # Test Diffusion solver
    print("\nTesting Diffusion solver...")
    diff_solver = DiffusionSolver2D(N, L, device)
    omega0_diff = diff_solver.taylor_green_vortex()
    
    diff_result = diff_solver.solve(omega0_diff, T, dt, nu=0.1, n_snapshots=10)
    print(f"Diffusion simulation completed. Output shape: {diff_result.shape}")
    print(f"Initial field range: [{omega0_diff.min():.3f}, {omega0_diff.max():.3f}]")
    print(f"Final field range: [{diff_result[-1].min():.3f}, {diff_result[-1].max():.3f}]")
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()
