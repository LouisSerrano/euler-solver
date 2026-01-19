import numpy as np
import torch
import torch.fft as fft
import matplotlib.pyplot as plt

# ============================================================
# Utility: periodic minimum-image displacement (crucial!)
# ============================================================
def periodic_delta(a, a0, L):
    d = a - a0
    return (d + 0.5 * L) % L - 0.5 * L


# ============================================================
# 2D Spectral Base Class (Real FFT)
# ============================================================
class Spectral2D:
    def __init__(self, N=128, L=2*np.pi, device="cpu"):
        self.N = N
        self.L = L
        self.device = device
        self.dx = L / N

        # Physical grid
        x = torch.linspace(0, L - self.dx, N, device=device)
        self.X, self.Y = torch.meshgrid(x, x, indexing="ij")

        # Wavenumbers (rfft2 shape)
        kx = 2*np.pi * torch.fft.fftfreq(N, d=self.dx).to(device)
        ky = 2*np.pi * torch.fft.rfftfreq(N, d=self.dx).to(device)
        self.KX, self.KY = torch.meshgrid(kx, ky, indexing="ij")

        self.k2 = self.KX**2 + self.KY**2
        self.k2_safe = self.k2.clone()
        self.k2_safe[0, 0] = 1.0

        self.ikx = 1j * self.KX
        self.iky = 1j * self.KY

        # 2/3 dealiasing mask
        kmax_global = (N // 2) * (2*np.pi / L)
        k_cut = (2.0 / 3.0) * kmax_global
        self.dealias = (torch.abs(self.KX) <= k_cut) & (torch.abs(self.KY) <= k_cut)

    def rfft2(self, f):
        return fft.rfft2(f)

    def irfft2(self, f_hat):
        return fft.irfft2(f_hat, s=(self.N, self.N))


# ============================================================
# Euler / Navier-Stokes in Vorticity Form
# ============================================================
class VorticitySolver2D(Spectral2D):
    def __init__(self, N=128, L=2*np.pi, device="cpu", filter_order=10, alpha=48.0):
        super().__init__(N, L, device)

        # Exponential spectral filter (damps near kmax)
        kmax = (N // 2) * (2*np.pi / L)
        kmag = torch.sqrt(self.k2)
        self.filter = torch.exp(-alpha * (kmag / kmax) ** filter_order).to(device)

    # -------------------------
    # Initial conditions
    # -------------------------
    def taylor_green(self, amp=1.0):
        # periodic and smooth
        return amp * (torch.sin(self.X) * torch.cos(self.Y) - torch.cos(self.X) * torch.sin(self.Y))

    def periodic_random_vortices(self, n_vortices=10, strength=8.0, sigma=None, seed=0):
        # Periodic Gaussian vortices (minimum image distance)
        if seed is not None:
            torch.manual_seed(seed)

        if sigma is None:
            sigma = max(3.0 * self.dx, self.L / 25)  # avoid too-sharp vortices

        omega = torch.zeros((self.N, self.N), device=self.device)
        for _ in range(n_vortices):
            x0 = torch.rand(1, device=self.device) * self.L
            y0 = torch.rand(1, device=self.device) * self.L

            sgn = (torch.randint(0, 2, (1,), device=self.device) * 2 - 1).float()
            amp = sgn * strength * (0.5 + torch.rand(1, device=self.device))

            dx = periodic_delta(self.X, x0, self.L)
            dy = periodic_delta(self.Y, y0, self.L)
            r2 = dx*dx + dy*dy

            omega += amp * torch.exp(-r2 / (2*sigma*sigma))

        return omega

    def vortex_dipole(self, strength=5.0, distance=None):
        """Two counter-rotating vortices (periodic)"""
        if distance is None:
            distance = self.L / 8
        
        x_c, y_c = self.L / 2, self.L / 2
        
        # Positive vortex
        dx1 = periodic_delta(self.X, x_c - distance/2, self.L)
        dy1 = periodic_delta(self.Y, y_c, self.L)
        r1_sq = dx1**2 + dy1**2
        
        # Negative vortex
        dx2 = periodic_delta(self.X, x_c + distance/2, self.L)
        dy2 = periodic_delta(self.Y, y_c, self.L)
        r2_sq = dx2**2 + dy2**2
        
        sigma = self.L / 25
        return strength * (torch.exp(-r1_sq/(2*sigma**2)) - torch.exp(-r2_sq/(2*sigma**2)))

    def vortex_merger(self, strength=5.0, distance=None):
        """Two same-sign vortices (periodic)"""
        if distance is None:
            distance = self.L / 6
            
        x_c, y_c = self.L / 2, self.L / 2
        
        dx1 = periodic_delta(self.X, x_c - distance/2, self.L)
        dy1 = periodic_delta(self.Y, y_c, self.L)
        r1_sq = dx1**2 + dy1**2
        
        dx2 = periodic_delta(self.X, x_c + distance/2, self.L)
        dy2 = periodic_delta(self.Y, y_c, self.L)
        r2_sq = dx2**2 + dy2**2
        
        sigma = self.L / 25
        return strength * (torch.exp(-r1_sq/(2*sigma**2)) + torch.exp(-r2_sq/(2*sigma**2)))

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
            
        omega = torch.zeros((self.N, self.N), device=self.device)
        # Using integers n, m to ensure periodicity on [0, L]
        for n in range(1, n_modes + 1):
            for m in range(1, n_modes + 1):
                kx = n * 2 * np.pi / self.L
                ky = m * 2 * np.pi / self.L
                
                # Random coefficients for a complex smooth field
                a = torch.randn(1, device=self.device) * amp / (n + m)
                phi = torch.rand(1, device=self.device) * 2 * np.pi
                psi = torch.rand(1, device=self.device) * 2 * np.pi
                
                omega += a * torch.sin(kx * self.X + phi) * torch.sin(ky * self.Y + psi)
                
        return omega

    # -------------------------
    # Physics helpers
    # -------------------------
    def velocity_from_vorticity_hat(self, omega_hat):
        # stream function: omega = -Δ psi -> psi_hat = -omega_hat / k^2
        psi_hat = -omega_hat / self.k2_safe.to(omega_hat.dtype)
        psi_hat[..., 0, 0] = 0.0

        u = self.irfft2(-self.iky * psi_hat)
        v = self.irfft2(self.ikx * psi_hat)
        return u, v

    def post_filter(self, omega):
        omega_hat = self.rfft2(omega)
        omega_hat = omega_hat * self.dealias * self.filter
        return self.irfft2(omega_hat)

    # -------------------------
    # Diagnostics: energy, enstrophy
    # -------------------------
    def energy_enstrophy(self, omega):
        """
        Energy = 1/2 ∫ |u|^2 dx
        Enstrophy = 1/2 ∫ omega^2 dx
        """
        omega_hat = self.rfft2(omega) * self.dealias
        u, v = self.velocity_from_vorticity_hat(omega_hat)

        E = 0.5 * torch.mean(u*u + v*v) * (self.L**2)
        Z = 0.5 * torch.mean(omega*omega) * (self.L**2)
        return E.item(), Z.item()

    # -------------------------
    # RHS and time stepping
    # -------------------------
    def rhs(self, omega, nu=0.0, nu_h=0.0, p=4):
        """
        omega_t + u·grad(omega) = nu Δ omega - nu_h (-Δ)^p omega
        """
        omega_hat = self.rfft2(omega)

        # Pre-clean state before nonlinear term
        omega_hat = omega_hat * self.dealias * self.filter

        # Compute velocity + gradients
        u, v = self.velocity_from_vorticity_hat(omega_hat)
        wx = self.irfft2(self.ikx * omega_hat)
        wy = self.irfft2(self.iky * omega_hat)

        # Nonlinear term
        adv = -(u * wx + v * wy)
        adv_hat = self.rfft2(adv)

        # Dealias the nonlinear product
        rhs_hat = adv_hat * self.dealias

        # Add viscosity (normal diffusion)
        if nu > 0:
            rhs_hat -= nu * self.k2 * omega_hat

        # Hyperviscosity (numerical sink for Euler stability)
        if nu_h > 0:
            rhs_hat -= nu_h * (self.k2 ** p) * omega_hat

        rhs = self.irfft2(rhs_hat)
        return rhs

    def rk4_step(self, omega, dt, nu=0.0, nu_h=0.0, p=4):
        k1 = self.rhs(omega, nu, nu_h, p)
        k2 = self.rhs(omega + 0.5*dt*k1, nu, nu_h, p)
        k3 = self.rhs(omega + 0.5*dt*k2, nu, nu_h, p)
        k4 = self.rhs(omega + dt*k3, nu, nu_h, p)
        omega_next = omega + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Post-filtering strongly reduces Gibbs + spectral blocking
        return self.post_filter(omega_next)

    def stable_dt(self, omega, cfl=0.4, nu=0.0):
        omega_hat = self.rfft2(omega) * self.dealias
        u, v = self.velocity_from_vorticity_hat(omega_hat)
        max_speed = torch.max(torch.sqrt(u*u + v*v)).item()

        # advective CFL limit
        if max_speed < 1e-12:
            dt_adv = 1.0
        else:
            dt_adv = cfl * self.dx / max_speed

        # explicit diffusion limit
        if nu > 1e-12:
            dt_diff = 0.5 * self.dx * self.dx / nu
        else:
            dt_diff = 1e9

        return min(dt_adv, dt_diff)

    def solve(self, omega0, T=2.0, n_snapshots=40, nu=0.0, nu_h=0.0, p=4, cfl=0.4):
        """
        Returns:
            times: (n_snapshots,)
            snapshots: (n_snapshots, N, N)
            energy: (n_snapshots,)
            enstrophy: (n_snapshots,)
        """
        omega = omega0.clone()
        omega = self.post_filter(omega)

        times = torch.linspace(0.0, T, n_snapshots, device=self.device)
        snapshots = torch.empty((n_snapshots, self.N, self.N), device=self.device)
        energy = torch.empty((n_snapshots,), device=self.device)
        enstrophy = torch.empty((n_snapshots,), device=self.device)

        # save initial
        snapshots[0] = omega
        E0, Z0 = self.energy_enstrophy(omega)
        energy[0] = E0
        enstrophy[0] = Z0

        t = 0.0
        for i in range(1, n_snapshots):
            t_target = times[i].item()

            # step exactly to snapshot time
            while t < t_target - 1e-12:
                dt = self.stable_dt(omega, cfl=cfl, nu=nu)
                dt = min(dt, t_target - t)
                dt = max(dt, 1e-8)  # floor
                omega = self.rk4_step(omega, dt, nu=nu, nu_h=nu_h, p=p)
                t += dt

            snapshots[i] = omega
            Ei, Zi = self.energy_enstrophy(omega)
            energy[i] = Ei
            enstrophy[i] = Zi

        return times.detach().cpu().numpy(), snapshots.detach().cpu().numpy(), energy.cpu().numpy(), enstrophy.cpu().numpy()


# ============================================================
# Plotting helpers
# ============================================================
def plot_snapshots_grid(snapshots, times, title, n_show=6, filename=None):
    """
    snapshots: (T, N, N)
    """
    T = snapshots.shape[0]
    idx = np.linspace(0, T-1, n_show, dtype=int)

    fig, axes = plt.subplots(1, n_show, figsize=(3.2*n_show, 3.0))
    fig.suptitle(title)

    vmin = np.min(snapshots[idx])
    vmax = np.max(snapshots[idx])

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

    # ----------------------------
    # Simulation parameters
    # ----------------------------
    N = 128
    L = 2*np.pi
    T = 10.0
    n_snap = 40

    # Filtering (Gibbs control)
    filter_order = 12
    alpha = 32 #48.0

    # Euler stabilization
    nu_euler = 0.0
    nu_h = 1e-13  # << try 1e-12..1e-8 depending on N/IC
    p = 4

    # Navier-Stokes
    nu_ns = 1e-3

    cfl = 0.45

    # ----------------------------
    # Setup solver + initial condition
    # ----------------------------
    solver = VorticitySolver2D(N=N, L=L, device=device, filter_order=filter_order, alpha=alpha)

    # Choose ONE initial condition
    # omega0 = solver.periodic_random_vortices(n_vortices=12, strength=8.0, sigma=None, seed=0)
    # omega0 = solver.taylor_green(amp=5.0)
    # omega0 = solver.vortex_dipole(strength=10.0)
    # omega0 = solver.vortex_merger(strength=10.0)
    # omega0 = solver.shear_layer(delta=0.1, U=2.0)
    omega0 = solver.sine_modes(n_modes=5, amp=10.0, seed=123)

    # ----------------------------
    # Run Euler
    # ----------------------------
    print("\nRunning Euler...")
    t_e, w_e, E_e, Z_e = solver.solve(
        omega0, T=T, n_snapshots=n_snap,
        nu=nu_euler, nu_h=nu_h, p=p, cfl=cfl
    )

    # ----------------------------
    # Run Navier-Stokes
    # ----------------------------
    print("\nRunning Navier-Stokes...")
    t_n, w_n, E_n, Z_n = solver.solve(
        omega0, T=T, n_snapshots=n_snap,
        nu=nu_ns, nu_h=0.0, p=p, cfl=cfl
    )

    # ----------------------------
    # Plot snapshots
    # ----------------------------
    plot_snapshots_grid(w_e, t_e, f"Euler Vorticity (N={N}, nu_h={nu_h}, p={p})", n_show=6, filename="euler_snapshots.png")
    plot_snapshots_grid(w_n, t_n, f"Navier-Stokes Vorticity (N={N}, nu={nu_ns})", n_show=6, filename="ns_snapshots.png")

    # ----------------------------
    # Plot energy/enstrophy
    # ----------------------------
    plot_energy_enstrophy(t_e, E_e, Z_e, E_n, Z_n, filename_E="energy_evolution.png", filename_Z="enstrophy_evolution.png")

    print("\nDone.")
    print("Euler final vorticity range:", np.min(w_e[-1]), np.max(w_e[-1]))
    print("NS final vorticity range:", np.min(w_n[-1]), np.max(w_n[-1]))


if __name__ == "__main__":
    main()

