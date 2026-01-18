import numpy as np
import torch
import torch.fft as fft
import matplotlib.pyplot as plt
import os
from datetime import datetime


# ============================================================
# Helpers
# ============================================================
def periodic_delta(a, a0, L):
    """Minimum-image displacement on periodic domain [0, L)."""
    d = a - a0
    return (d + 0.5 * L) % L - 0.5 * L


def fftshift2(x):
    return fft.fftshift(x, dim=(-2, -1))


def ifftshift2(x):
    return fft.ifftshift(x, dim=(-2, -1))


def downsample_spectral(omega, target_N):
    """
    Downsample a physical field 'omega' (NxN) to 'target_N x target_N'
    using spectral truncation.
    """
    N = omega.shape[-1]
    if target_N == N:
        return omega

    # 1. To spectral
    omega_hat = fft.fft2(omega)
    omega_hat_shift = fftshift2(omega_hat)

    # 2. Truncate
    s = (N - target_N) // 2
    omega_hat_trunc_shift = omega_hat_shift[..., s:s+target_N, s:s+target_N]
    omega_hat_trunc = ifftshift2(omega_hat_trunc_shift)

    # 3. Back to physical
    # Scale by (target_N/N)^2 to preserve amplitudes (due to FFT normalization differences)
    # Actually, ifft2(fft2(x)) = x. Truncation reduces the number of modes.
    # The amplitude in physical space is preserved naturally if we use the same L.
    # But we need to account for the normalization factor of the smaller FFT.
    # fft2(x) has sum(x), ifft2(X) has 1/N^2 * sum(X * exp).
    # so we need to multiply by (target_N/N)^2
    omega_low = fft.ifft2(omega_hat_trunc).real * (target_N / N)**2
    return omega_low


# ============================================================
# 2D Pseudo-spectral solver with 3/2 padding dealiasing
# Vorticity form:  omega_t + u·grad omega = nu Δ omega
# with u = ( -psi_y, psi_x ),  -Δ psi = omega
# ============================================================
class VorticitySpectral2D:
    """
    Uses full complex FFTs (fft2/ifft2) to make 3/2 padding simpler and robust.

    For Euler enstrophy conservation:
      - use float64 / complex128
      - use 3/2 padding for the nonlinear term
      - use implicit midpoint for time stepping
      - do NOT use filters/hyperviscosity
    """

    def __init__(self, N=128, L=2*np.pi, device="cpu"):
        assert (3 * N) % 2 == 0, "N must be even for 3/2 padding to be integer."
        self.N = int(N)
        self.L = float(L)
        self.device = device

        # Promote numeric stability/invariants
        self.real_dtype = torch.float64
        self.cplx_dtype = torch.complex128

        self.dx = self.L / self.N

        # Physical grid (N x N)
        x = torch.linspace(0.0, self.L - self.dx, self.N, device=device, dtype=self.real_dtype)
        self.X, self.Y = torch.meshgrid(x, x, indexing="ij")

        # Padded grid size (3/2 rule)
        self.M = int(3 * self.N // 2)
        self.dxM = self.L / self.M
        xM = torch.linspace(0.0, self.L - self.dxM, self.M, device=device, dtype=self.real_dtype)
        self.XM, self.YM = torch.meshgrid(xM, xM, indexing="ij")

        # Wavenumbers for N grid (full fft2)
        kxN = 2*np.pi * torch.fft.fftfreq(self.N, d=self.dx).to(device=device, dtype=self.real_dtype)
        kyN = 2*np.pi * torch.fft.fftfreq(self.N, d=self.dx).to(device=device, dtype=self.real_dtype)
        self.KXN, self.KYN = torch.meshgrid(kxN, kyN, indexing="ij")
        self.k2N = self.KXN**2 + self.KYN**2
        self.k2N_safe = self.k2N.clone()
        self.k2N_safe[0, 0] = 1.0

        self.ikxN = (1j * self.KXN).to(self.cplx_dtype)
        self.ikyN = (1j * self.KYN).to(self.cplx_dtype)

        # Wavenumbers for M padded grid
        kxM = 2*np.pi * torch.fft.fftfreq(self.M, d=self.dxM).to(device=device, dtype=self.real_dtype)
        kyM = 2*np.pi * torch.fft.fftfreq(self.M, d=self.dxM).to(device=device, dtype=self.real_dtype)
        self.KXM, self.KYM = torch.meshgrid(kxM, kyM, indexing="ij")
        self.k2M = self.KXM**2 + self.KYM**2
        self.k2M_safe = self.k2M.clone()
        self.k2M_safe[0, 0] = 1.0

        self.ikxM = (1j * self.KXM).to(self.cplx_dtype)
        self.ikyM = (1j * self.KYM).to(self.cplx_dtype)

    # ----------------------------
    # FFT wrappers
    # ----------------------------
    def fft2(self, f):
        return fft.fft2(f.to(self.real_dtype)).to(self.cplx_dtype)

    def ifft2(self, f_hat):
        return fft.ifft2(f_hat.to(self.cplx_dtype)).real.to(self.real_dtype)

    # ----------------------------
    # Padding/truncation (3/2 rule)
    # Uses centered spectrum (fftshift) to avoid index fiddling.
    # ----------------------------
    def pad_hat_N_to_M(self, f_hat_N):
        """
        Embed NxN spectrum into MxM spectrum by zero-padding high modes.
        """
        fN = fftshift2(f_hat_N)
        fM = torch.zeros((self.M, self.M), device=self.device, dtype=self.cplx_dtype)

        s = (self.M - self.N) // 2
        fM[s:s+self.N, s:s+self.N] = fN
        return ifftshift2(fM)

    def trunc_hat_M_to_N(self, f_hat_M):
        """
        Extract the central NxN block from MxM spectrum (inverse of padding).
        """
        fM = fftshift2(f_hat_M)
        s = (self.M - self.N) // 2
        fN = fM[s:s+self.N, s:s+self.N]
        return ifftshift2(fN)

    # ----------------------------
    # Velocity from vorticity in spectral space
    # ----------------------------
    def velocity_from_omega_hat_N(self, omega_hat_N):
        psi_hat = -omega_hat_N / self.k2N_safe.to(self.cplx_dtype)
        psi_hat[0, 0] = 0.0 + 0.0j
        u = self.ifft2(-self.ikyN * psi_hat)
        v = self.ifft2(self.ikxN * psi_hat)
        return u, v

    # ----------------------------
    # Nonlinear RHS with 3/2 de-aliasing
    # ----------------------------
    def rhs_hat_N(self, omega, nu=0.0):
        """
        Returns RHS in spectral space (NxN):
          omega_t_hat = -FFT( u·grad omega ) + nu * (-k^2) omega_hat
        Nonlinear term is computed with 3/2 padding dealiasing.
        """
        omega = omega.to(self.real_dtype)
        omega_hat_N = self.fft2(omega)  # NxN

        # --- Dealiased nonlinear term via 3/2 padding ---
        omega_hat_M = self.pad_hat_N_to_M(omega_hat_N)

        # streamfunction on padded grid
        psi_hat_M = -omega_hat_M / self.k2M_safe.to(self.cplx_dtype)
        psi_hat_M[0, 0] = 0.0 + 0.0j

        # velocities and gradients on padded grid
        uM = self.ifft2(-self.ikyM * psi_hat_M)
        vM = self.ifft2(self.ikxM * psi_hat_M)
        wxM = self.ifft2(self.ikxM * omega_hat_M)
        wyM = self.ifft2(self.ikyM * omega_hat_M)

        advM = -(uM * wxM + vM * wyM)  # Jacobian form equivalent

        adv_hat_M = self.fft2(advM)
        adv_hat_N = self.trunc_hat_M_to_N(adv_hat_M)

        rhs_hat = adv_hat_N

        # diffusion (Navier–Stokes)
        if nu > 0.0:
            rhs_hat = rhs_hat - nu * (self.k2N.to(self.cplx_dtype)) * omega_hat_N

        return rhs_hat

    def rhs(self, omega, nu=0.0):
        """Return RHS in physical space."""
        return self.ifft2(self.rhs_hat_N(omega, nu=nu))

    # ----------------------------
    # Diagnostics (physical-space integrals)
    # ----------------------------
    def energy_enstrophy(self, omega):
        """
        Energy:    E = 1/2 ∫ |u|^2 dx
        Enstrophy: Z = 1/2 ∫ omega^2 dx
        """
        omega = omega.to(self.real_dtype)
        omega_hat = self.fft2(omega)
        u, v = self.velocity_from_omega_hat_N(omega_hat)

        dA = (self.dx * self.dx)
        E = 0.5 * dA * torch.sum(u*u + v*v)
        Z = 0.5 * dA * torch.sum(omega*omega)
        return float(E.item()), float(Z.item())

    # ----------------------------
    # Time stepping: implicit midpoint (fixed-point iteration)
    # ----------------------------
    def step_midpoint(self, omega_n, dt, nu=0.0, n_iter=6, tol=1e-12):
        """
        Implicit midpoint:
          omega_{n+1} = omega_n + dt * RHS( (omega_n + omega_{n+1})/2 )

        Fixed-point iterations:
          omega_{n+1}^{(m+1)} = omega_n + dt * RHS( (omega_n + omega_{n+1}^{(m)})/2 )
        """
        omega_n = omega_n.to(self.real_dtype)
        omega_new = omega_n.clone()

        for _ in range(n_iter):
            omega_mid = 0.5 * (omega_n + omega_new)
            rhs_mid = self.rhs(omega_mid, nu=nu)
            omega_next = omega_n + dt * rhs_mid

            # convergence check
            err = torch.max(torch.abs(omega_next - omega_new)).item()
            omega_new = omega_next
            if err < tol:
                break

        return omega_new

    # ----------------------------
    # Stable dt (CFL + diffusion)
    # ----------------------------
    def stable_dt(self, omega, cfl=0.3, nu=0.0):
        omega = omega.to(self.real_dtype)
        omega_hat = self.fft2(omega)
        u, v = self.velocity_from_omega_hat_N(omega_hat)
        speed = torch.sqrt(u*u + v*v)
        umax = float(torch.max(speed).item())

        if umax < 1e-14:
            dt_adv = 1.0
        else:
            dt_adv = cfl * self.dx / umax

        if nu > 1e-14:
            dt_diff = 0.5 * self.dx * self.dx / nu
        else:
            dt_diff = 1e30

        return min(dt_adv, dt_diff)

    # ----------------------------
    # Solve and record snapshots
    # ----------------------------
    def solve(self, omega0, T=2.0, n_snapshots=40, nu=0.0, cfl=0.3, n_iter_midpoint=6):
        omega = omega0.to(self.device, dtype=self.real_dtype).clone()

        times = np.linspace(0.0, T, n_snapshots)
        snapshots = np.zeros((n_snapshots, self.N, self.N), dtype=np.float64)
        E = np.zeros((n_snapshots,), dtype=np.float64)
        Z = np.zeros((n_snapshots,), dtype=np.float64)

        snapshots[0] = omega.detach().cpu().numpy()
        E[0], Z[0] = self.energy_enstrophy(omega)

        t = 0.0
        for i in range(1, n_snapshots):
            t_target = times[i]
            while t < t_target - 1e-15:
                dt = self.stable_dt(omega, cfl=cfl, nu=nu)
                dt = min(dt, t_target - t)
                dt = max(dt, 1e-10)
                omega = self.step_midpoint(omega, dt, nu=nu, n_iter=n_iter_midpoint)
                t += dt

            snapshots[i] = omega.detach().cpu().numpy()
            E[i], Z[i] = self.energy_enstrophy(omega)

        return times, snapshots, E, Z

    # ----------------------------
    # Initial conditions (periodic)
    # ----------------------------
    def taylor_green(self, amp=5.0):
        # smooth + periodic
        return amp * (torch.sin(self.X) * torch.cos(self.Y) - torch.cos(self.X) * torch.sin(self.Y))

    def periodic_random_vortices(self, n_vortices=12, strength=8.0, sigma=None, seed=0):
        if seed is not None:
            torch.manual_seed(seed)

        if sigma is None:
            sigma = max(4.0 * self.dx, self.L / 25.0)  # avoid too-sharp initial vorticity

        omega = torch.zeros((self.N, self.N), device=self.device, dtype=self.real_dtype)
        for _ in range(n_vortices):
            x0 = torch.rand(1, device=self.device, dtype=self.real_dtype) * self.L
            y0 = torch.rand(1, device=self.device, dtype=self.real_dtype) * self.L

            sgn = (torch.randint(0, 2, (1,), device=self.device) * 2 - 1).to(self.real_dtype)
            amp = sgn * strength * (0.5 + torch.rand(1, device=self.device, dtype=self.real_dtype))

            dx = periodic_delta(self.X, x0, self.L)
            dy = periodic_delta(self.Y, y0, self.L)
            r2 = dx*dx + dy*dy

            omega += amp * torch.exp(-r2 / (2.0 * sigma * sigma))

        return omega

    def shear_layer(self, delta=0.1, U=1.0, perturbation=0.05):
        """Standard periodic shear layer"""
        # Two layers to maintain periodicity in y
        omega = U/delta * (1.0/torch.cosh((self.Y - self.L/4)/delta)**2 - 
                         1.0/torch.cosh((3*self.L/4 - self.Y)/delta)**2)
        
        # Add sinusoidal perturbation
        perturb = perturbation * U * torch.sin(2 * np.pi * self.X / self.L)
        return omega + perturb

    def sine_modes(self, n_modes=4, amp=5.0, seed=42):
        """Sum of low-frequency sine/cosine modes with correct periodicity"""
        if seed is not None:
            torch.manual_seed(seed)
            
        omega = torch.zeros((self.N, self.N), device=self.device, dtype=self.real_dtype)
        # Using integers n, m to ensure periodicity on [0, L]
        for n in range(1, n_modes + 1):
            for m in range(1, n_modes + 1):
                kx = n * 2 * np.pi / self.L
                ky = m * 2 * np.pi / self.L
                
                # Random coefficients for a complex smooth field
                a = torch.randn(1, device=self.device, dtype=self.real_dtype) * amp / (n + m)
                phi = torch.rand(1, device=self.device, dtype=self.real_dtype) * 2 * np.pi
                psi = torch.rand(1, device=self.device, dtype=self.real_dtype) * 2 * np.pi
                
                omega += a * torch.sin(kx * self.X + phi) * torch.sin(ky * self.Y + psi)
                
        return omega


# ============================================================
# Plot helpers
# ============================================================
def plot_snapshots_grid(snapshots, times, title, n_show=6, filename=None):
    T = snapshots.shape[0]
    idx = np.linspace(0, T-1, n_show, dtype=int)

    fig, axes = plt.subplots(1, n_show, figsize=(3.2*n_show, 3.0))
    fig.suptitle(title)

    vmin = np.min(snapshots[idx])
    vmax = np.max(snapshots[idx])

    im = None
    for ax, j in zip(axes, idx):
        im = ax.imshow(snapshots[j], origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"t={times[j]:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    plt.show()


def plot_energy_enstrophy(times, E_euler, Z_euler, E_ns, Z_ns, filename_E=None, filename_Z=None):
    plt.figure(figsize=(6, 4))
    plt.plot(times, E_euler, label="Euler Energy")
    plt.plot(times, E_ns, label="NS Energy")
    plt.xlabel("time")
    plt.ylabel("Energy")
    plt.title("Energy vs Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if filename_E:
        plt.savefig(filename_E, dpi=150, bbox_inches='tight')
        print(f"Energy plot saved to {filename_E}")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(times, Z_euler, label="Euler Enstrophy")
    plt.plot(times, Z_ns, label="NS Enstrophy")
    plt.xlabel("time")
    plt.ylabel("Enstrophy")
    plt.title("Enstrophy vs Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if filename_Z:
        plt.savefig(filename_Z, dpi=150, bbox_inches='tight')
        print(f"Enstrophy plot saved to {filename_Z}")
    plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Output directory setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/run_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving results to: {results_dir}")

    N_high = 512
    L = 2*np.pi
    T = 10.0 # Reduced for faster execution in main
    n_snap = 10

    cfl = 0.35
    n_iter_mid = 6

    # Viscosities to compare
    viscosities = [0.0, 1e-4, 1e-3, 1e-2]
    final_snapshots = []
    labels = []

    # Initialize solver for high resolution
    solver = VorticitySpectral2D(N=N_high, L=L, device=device)
    omega0 = solver.sine_modes(n_modes=5, amp=10.0, seed=123)

    print(f"\nComparing {len(viscosities)} viscosities at N={N_high}...")
    
    for nu in viscosities:
        nu_label = "Euler" if nu == 0.0 else f"nu={nu}"
        print(f"\n--- Running {nu_label} ---")
        t, w, E, Z = solver.solve(
            omega0, T=T, n_snapshots=n_snap, nu=nu, cfl=cfl, n_iter_midpoint=n_iter_mid
        )
        
        # Save trajectory plot (just to keep track)
        traj_filename = os.path.join(results_dir, f"snapshots_{nu_label.replace('=', '_')}.png")
        plot_snapshots_grid(w, t, f"Vorticity ({nu_label}, N={N_high})", n_show=5, filename=traj_filename)
        
        final_snapshots.append(w[-1])
        labels.append(nu_label)

    # 1. Viscosity Comparison Plot
    fig, axes = plt.subplots(1, len(viscosities), figsize=(4*len(viscosities), 4))
    fig.suptitle(f"Viscosity Comparison at T={T:.1f} (N={N_high})")
    for i, (snap, label) in enumerate(zip(final_snapshots, labels)):
        im = axes[i].imshow(snap, origin="lower")
        axes[i].set_title(label)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    visc_comp_path = os.path.join(results_dir, "viscosity_comparison.png")
    plt.savefig(visc_comp_path, dpi=150, bbox_inches='tight')
    print(f"Viscosity comparison saved to {visc_comp_path}")
    plt.close()

    # 2. Resolution Comparison Plot (using the Euler result)
    print("\nGenerating downsampled resolution plots...")
    euler_last = torch.from_numpy(final_snapshots[0]).to(device)
    resolutions = [512, 256, 128]
    downsampled_fields = []

    for res in resolutions:
        # Downsample
        w_low = downsample_spectral(euler_last, res).cpu().numpy()
        downsampled_fields.append(w_low)
        
        # Save individual plot
        plt.figure(figsize=(5, 5))
        plt.imshow(w_low, origin="lower")
        plt.title(f"Vorticity {res}x{res} (Downsampled)")
        plt.axis('off')
        res_plot_path = os.path.join(results_dir, f"vorticity_{res}.png")
        plt.savefig(res_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {res}x{res} plot to {res_plot_path}")

    # Side-by-side comparison
    fig, axes = plt.subplots(1, len(resolutions), figsize=(4*len(resolutions), 4))
    fig.suptitle(f"Resolution Comparison (Euler at T={T:.1f})")
    for i, (snap, res) in enumerate(zip(downsampled_fields, resolutions)):
        im = axes[i].imshow(snap, origin="lower")
        axes[i].set_title(f"{res}x{res}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    res_comp_path = os.path.join(results_dir, "resolution_comparison.png")
    plt.savefig(res_comp_path, dpi=150, bbox_inches='tight')
    print(f"Resolution comparison saved to {res_comp_path}")
    plt.close()

    print("\nResults saved in:", results_dir)


if __name__ == "__main__":
    main()

