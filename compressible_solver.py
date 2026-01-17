
import torch
import torch.nn.functional as F
import numpy as np

class CompressibleSolver:
    """
    Compressible Euler/Navier-Stokes Solver using Finite Volume Method.
    Supports 2D simulation with Periodic Boundary Conditions.
    """
    def __init__(self, nx=256, ny=256, lx=1.0, ly=1.0, gamma=1.4, bc_mode='circular', device='cuda'):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.dx = lx / nx
        self.dy = ly / ny
        self.gamma = gamma
        self.bc_mode = bc_mode
        self.device = device

        # Grid coordinates (cell centers)
        x = torch.linspace(self.dx/2, self.lx - self.dx/2, nx, device=device)
        y = torch.linspace(self.dy/2, self.ly - self.dy/2, ny, device=device)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')

    def get_primitive(self, Q):
        """
        Convert conservative variables Q to primitive variables.
        Q: [Batch, 4, Nx, Ny] -> (rho, rho*u, rho*v, E)
        Returns: rho, u, v, p
        """
        rho = Q[:, 0]
        # Positivity preservation: clamp density to avoid division by zero and negative values
        rho = torch.clamp(rho, min=1e-6)
        
        u = Q[:, 1] / rho
        v = Q[:, 2] / rho
        E = Q[:, 3]
        
        # p = (gamma - 1) * (E - 0.5 * rho * (u^2 + v^2))
        kinetic_energy = 0.5 * rho * (u**2 + v**2)
        p = (self.gamma - 1) * (E - kinetic_energy)
        
        # Positivity preservation: clamp pressure to avoid NaN in sound speed calculations
        p = torch.clamp(p, min=1e-6)
        
        return rho, u, v, p

    def get_conservative(self, rho, u, v, p):
        """Convert primitive variables to conservative variables Q."""
        rho_u = rho * u
        rho_v = rho * v
        kinetic_energy = 0.5 * rho * (u**2 + v**2)
        E = p / (self.gamma - 1) + kinetic_energy
        
        return torch.stack([rho, rho_u, rho_v, E], dim=1)

    def minmod(self, a, b):
        """
        MinMod limiter for slope limiting in MUSCL reconstruction.
        Returns sign(a) * max(0, min(|a|, sign(a)*b))
        """
        return torch.where(
            a * b <= 0,
            torch.zeros_like(a),
            torch.where(
                torch.abs(a) < torch.abs(b),
                a,
                b
            )
        )

    def compute_slopes(self, Q_pad, dim):
        """
        Compute limited slopes for MUSCL reconstruction.
        Q_pad: Padded conservative variables
        dim: 1 for x-direction, 2 for y-direction
        Returns: Limited slopes at cell centers in the active domain
        """
        if dim == 1:
            # X-direction slopes: compute for cells that have neighbors
            # We need slopes for the active domain cells
            Q_minus = Q_pad[:, :, :-2, :]  # Q_{i-1}
            Q_center = Q_pad[:, :, 1:-1, :]  # Q_i
            Q_plus = Q_pad[:, :, 2:, :]  # Q_{i+1}
        else:
            # Y-direction slopes
            Q_minus = Q_pad[:, :, :, :-2]  # Q_{j-1}
            Q_center = Q_pad[:, :, :, 1:-1]  # Q_j
            Q_plus = Q_pad[:, :, :, 2:]  # Q_{j+1}
        
        # Backward and forward differences
        delta_minus = Q_center - Q_minus
        delta_plus = Q_plus - Q_center
        
        # Apply MinMod limiter component-wise
        slopes = self.minmod(delta_minus, delta_plus)
        
        return slopes

    def fluxes_explicit(self, Q):
        """
        Compute Convective Fluxes using HLLC Riemann Solver with MUSCL reconstruction.
        2nd-order accurate in space with MinMod slope limiting on primitive variables.
        """
        # Input Q: [B, 4, nx, ny] where nx=ny=128
        # Need to return: [B, 4, nx, ny]
        
        # Padding for BCs - need 2 ghost cells for MUSCL stencil
        pad = 2
        Q_pad = F.pad(Q, (pad, pad, pad, pad), mode=self.bc_mode)
        # Q_pad: [B, 4, 132, 132]
        
        # Convert to primitive variables for limiting (more robust for shocks)
        rho_pad, u_pad, v_pad, p_pad = self.get_primitive(Q_pad)
        
        # Stack primitives
        W_pad = torch.stack([rho_pad, u_pad, v_pad, p_pad], dim=1)
        
        # --- MUSCL Reconstruction for X-Direction ---
        slopes_x = self.compute_slopes(W_pad, dim=1)
        # slopes_x: [B, 4, 130, 132] (130 cells in X, all cells in Y)
        
        # Extract the region needed for active domain
        W_cells_x = W_pad[:, :, 1:-1, 2:-2]  # [B, 4, 130, 128]
        slopes_x_trimmed = slopes_x[:, :, :, 2:-2]  # [B, 4, 130, 128]
        
        # Reconstruct primitive variables at interfaces
        W_L_x = W_cells_x[:, :, :-1, :] + 0.5 * slopes_x_trimmed[:, :, :-1, :]  # [B, 4, 129, 128]
        W_R_x = W_cells_x[:, :, 1:, :] - 0.5 * slopes_x_trimmed[:, :, 1:, :]    # [B, 4, 129, 128]
        
        # Ensure positivity of reconstructed primitives
        W_L_x[:, 0] = torch.clamp(W_L_x[:, 0], min=1e-6)  # rho
        W_L_x[:, 3] = torch.clamp(W_L_x[:, 3], min=1e-6)  # p
        W_R_x[:, 0] = torch.clamp(W_R_x[:, 0], min=1e-6)
        W_R_x[:, 3] = torch.clamp(W_R_x[:, 3], min=1e-6)
        
        # Convert back to conservative
        Q_L_x = self.get_conservative(W_L_x[:, 0], W_L_x[:, 1], W_L_x[:, 2], W_L_x[:, 3])
        Q_R_x = self.get_conservative(W_R_x[:, 0], W_R_x[:, 1], W_R_x[:, 2], W_R_x[:, 3])
        
        Flux_X = self.hllc_flux(Q_L_x, Q_R_x, dim=1)  # [B, 4, 129, 128]
        
        # --- MUSCL Reconstruction for Y-Direction ---
        slopes_y = self.compute_slopes(W_pad, dim=2)
        # slopes_y: [B, 4, 132, 130]
        
        W_cells_y = W_pad[:, :, 2:-2, 1:-1]  # [B, 4, 128, 130]
        slopes_y_trimmed = slopes_y[:, :, 2:-2, :]  # [B, 4, 128, 130]
        
        W_L_y = W_cells_y[:, :, :, :-1] + 0.5 * slopes_y_trimmed[:, :, :, :-1]  # [B, 4, 128, 129]
        W_R_y = W_cells_y[:, :, :, 1:] - 0.5 * slopes_y_trimmed[:, :, :, 1:]    # [B, 4, 128, 129]
        
        # Ensure positivity
        W_L_y[:, 0] = torch.clamp(W_L_y[:, 0], min=1e-6)
        W_L_y[:, 3] = torch.clamp(W_L_y[:, 3], min=1e-6)
        W_R_y[:, 0] = torch.clamp(W_R_y[:, 0], min=1e-6)
        W_R_y[:, 3] = torch.clamp(W_R_y[:, 3], min=1e-6)
        
        Q_L_y = self.get_conservative(W_L_y[:, 0], W_L_y[:, 1], W_L_y[:, 2], W_L_y[:, 3])
        Q_R_y = self.get_conservative(W_R_y[:, 0], W_R_y[:, 1], W_R_y[:, 2], W_R_y[:, 3])
        
        Flux_Y = self.hllc_flux(Q_L_y, Q_R_y, dim=2)  # [B, 4, 128, 129]
        
        # Divergence: (F_{i+1/2} - F_{i-1/2}) / dx
        dFdx = (Flux_X[:, :, 1:, :] - Flux_X[:, :, :-1, :]) / self.dx  # [B, 4, 128, 128]
        dGdy = (Flux_Y[:, :, :, 1:] - Flux_Y[:, :, :, :-1]) / self.dy  # [B, 4, 128, 128]
        
        return -(dFdx + dGdy)  # [B, 4, 128, 128]



    def hllc_flux(self, Q_L, Q_R, dim):
        """
        HLLC Riemann Solver.
        dim: 1 for X-flux, 2 for Y-flux.
        """
        # 1. Get Primitives
        rho_L, u_L, v_L, p_L = self.get_primitive(Q_L)
        rho_R, u_R, v_R, p_R = self.get_primitive(Q_R)
        
        # Normal velocity for the dimension
        if dim == 1:
            vn_L, vn_R = u_L, u_R
            vt_L, vt_R = v_L, v_R
        else:
            vn_L, vn_R = v_L, v_R
            vt_L, vt_R = u_L, u_R
            
        # 2. Sound Speeds
        c_L = torch.sqrt(self.gamma * p_L / (rho_L + 1e-8))
        c_R = torch.sqrt(self.gamma * p_R / (rho_R + 1e-8))
        
        # 3. Wave Speed Estimator (Davis / Roe-like bounds)
        # Using simple min/max bounds
        # S_L = min(vn_L - c_L, vn_R - c_R) # Robust
        # S_R = max(vn_L + c_L, vn_R + c_R)
        
        # Roe Averages might be better, but let's stick to Davis for simplicity of implementation
        s_min_L = vn_L - c_L
        s_min_R = vn_R - c_R
        S_L = torch.minimum(s_min_L, s_min_R)
        
        s_max_L = vn_L + c_L
        s_max_R = vn_R + c_R
        S_R = torch.maximum(s_max_L, s_max_R)
        
        # 4. Contact Wave Speed S_star
        # S_star = (p_R - p_L + rho_L*vn_L*(S_L - vn_L) - rho_R*vn_R*(S_R - vn_R)) 
        #          / (rho_L*(S_L - vn_L) - rho_R*(S_R - vn_R))
        
        num = p_R - p_L + rho_L * vn_L * (S_L - vn_L) - rho_R * vn_R * (S_R - vn_R)
        den = rho_L * (S_L - vn_L) - rho_R * (S_R - vn_R) + 1e-8
        S_star = num / den
        
        # 5. Compute Fluxes
        # Helper for physical flux
        def get_phys_flux(rho, vn, vt, p, E):
            # Flux in direction n: [rho*vn, rho*vn^2 + p, rho*vn*vt, (E+p)*vn]
            # Map back to x,y coords
            if dim == 1:
                return torch.stack([rho*vn, rho*vn**2 + p, rho*vn*vt, (E+p)*vn], dim=1)
            else:
                return torch.stack([rho*vn, rho*vt*vn, rho*vn**2 + p, (E+p)*vn], dim=1)

        E_L = Q_L[:, 3]
        E_R = Q_R[:, 3]
        F_L = get_phys_flux(rho_L, vn_L, vt_L, p_L, E_L)
        F_R = get_phys_flux(rho_R, vn_R, vt_R, p_R, E_R)
        
        # 6. HLLC Logic
        # Masking for regions
        # Region 1: S_L > 0 (Supersonic Left) -> F_L
        # Region 2: S_L <= 0 < S_star (Subsonic Left) -> F_star_L
        # Region 3: S_star <= 0 < S_R (Subsonic Right) -> F_star_R
        # Region 4: S_R < 0 (Supersonic Right) -> F_R
        
        # Shared factors for Star Fluxes
        # F_star_K = F_K + S_K * (Q_star_K - Q_K)
        
        def get_Q_star(rho, vn, vt, p, E, S_K):
            # Using limit to avoid div-by-zero
            denom = S_K - S_star + 1e-8
            factor = rho * (S_K - vn) / denom
            
            # Components
            comp_rho = torch.ones_like(rho)
            
            # Momentum
            if dim == 1:
                comp_u = S_star
                comp_v = vt
            else:
                comp_u = vt
                comp_v = S_star
                
            # Energy
            term2 = (S_star - vn) * (S_star + p / (rho * (S_K - vn) + 1e-8))
            comp_E = E / (rho + 1e-8) + term2
            
            # Resulting Q_star vector
            return factor.unsqueeze(1) * torch.stack([comp_rho, comp_u, comp_v, comp_E], dim=1)

        # Left Star
        Q_star_L = get_Q_star(rho_L, vn_L, vt_L, p_L, E_L, S_L)
        F_star_L = F_L + S_L.unsqueeze(1) * (Q_star_L - Q_L)
        
        # Right Star
        Q_star_R = get_Q_star(rho_R, vn_R, vt_R, p_R, E_R, S_R)
        F_star_R = F_R + S_R.unsqueeze(1) * (Q_star_R - Q_R)
        
        # Apply Masks with torch.where for proper broadcasting
        # Masks are [B, H, W]. Fluxes are [B, 4, H, W].
        mask_L = (S_L >= 0).unsqueeze(1)
        mask_R = (S_R <= 0).unsqueeze(1)
        # S_star needs unsqueeze for comparison? No, S_star has shape of S_L [B, H, W]
        # But we need unsqueezed masks for Flux selection.
        
        S_star_unsq = S_star.unsqueeze(1)
        mask_star_L = (~mask_L) & (~mask_R) & (S_star_unsq >= 0)
        # Remaining is mask_star_R
        
        # Logic: 
        # If mask_L: F_L
        # Else if mask_R: F_R
        # Else if mask_star_L: F_star_L
        # Else: F_star_R
        
        Flux = torch.where(mask_L, F_L,
                   torch.where(mask_R, F_R,
                       torch.where(mask_star_L, F_star_L, F_star_R)
                   )
               )
        
        return Flux

    def compute_viscous_dissipation(self, Q, mu):
        """Compute physical viscous dissipation rate (Phi)."""
        pad = 1
        Q_pad = F.pad(Q, (pad, pad, pad, pad), mode=self.bc_mode)
        rho, u, v, p = self.get_primitive(Q_pad)
        
        # Velocity gradients
        u_x = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * self.dx)
        u_y = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * self.dy)
        v_x = (v[:, 2:, 1:-1] - v[:, :-2, 1:-1]) / (2 * self.dx)
        v_y = (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) / (2 * self.dy)
        
        div_u = u_x + v_y
        
        # Strain rate tensor
        S_xx = u_x - (1.0/3.0) * div_u
        S_yy = v_y - (1.0/3.0) * div_u
        S_xy = 0.5 * (u_y + v_x)
        
        # Dissipation: Phi = 2mu * (S:S)
        phi = 2 * mu * (S_xx**2 + S_yy**2 + 2 * S_xy**2)
        
        return (phi * self.dx * self.dy).sum().item()

    def viscous_fluxes_explicit(self, Q, mu_visc):
        """
        Compute full viscous fluxes using Compact Stencil (Half-step gradients).
        Computes Fluxes at cell interfaces (i+1/2, j+1/2) for standard 3-point Laplacian accuracy.
        """
        if mu_visc == 0.0:
            return torch.zeros_like(Q)
        
        # Pad Q by 1 for cell centers
        pad = 1
        Q_pad = F.pad(Q, (pad, pad, pad, pad), mode=self.bc_mode)
        rho, u, v, p = self.get_primitive(Q_pad)
        
        # Constants
        dx, dy = self.dx, self.dy
        Pr = 0.72
        Cp = self.gamma / (self.gamma - 1.0)
        
        # We need fluxes at Interfaces.
        # X-Interfaces: (i+1/2, j). Indices correspond to [1:-2, 1:-1] of Q_pad?
        # Q_pad is [0...Nx+1]. Center [1...Nx].
        # Interfaces: i+1/2 for i=0..Nx. count is Nx+1.
        
        # --- 1. Compute Gradients and Properties at X-Interfaces (i+1/2) ---
        # Stencil: u_{i+1} - u_i
        # Range: Include ghosts for cross-terms?
        # Let's compute only for internal interfaces needed for Divergence?
        # Divergence at i needs F_{i+1/2} and F_{i-1/2}.
        # So we need Fluxes at all internal faces + boundary faces. 0.5 to Nx-0.5.
        
        # Helper averages to interfaces
        def avg_x(f): return 0.5 * (f[:, 1:] + f[:, :-1]) 
        def avg_y(f): return 0.5 * (f[:, :, 1:] + f[:, :, :-1])
        def diff_x(f): return (f[:, 1:] - f[:, :-1]) / dx
        def diff_y(f): return (f[:, :, 1:] - f[:, :, :-1]) / dy
        
        # We need full padded domain arrays to compute subsets
        # u is [B, Nx+2, Ny+2]
        
        # --- Flux X (at i+1/2) ---
        # Needs: u_x, v_x, u_y, v_y, dT_dx
        # u_x at i+1/2 is compact diff
        # u_y at i+1/2 is avg of centered y-diffs
        
        # X-derivatives at X-interfaces (Nx+1, Ny+2) -> trim Y later
        u_x_face = diff_x(u) # [Nx+1, Ny+2]
        v_x_face = diff_x(v)
        T = p / (rho + 1e-8)
        dT_dx_face = diff_x(T)
        
        # Y-derivatives at cell centers [Nx+2, Ny] -> avg to X-interfaces
        # Y-diff at centers: (u_{j+1} - u_{j-1})/2dy
        u_y_center = (u[:, :, 2:] - u[:, :, :-2]) / (2*dy) # [Nx+2, Ny]
        v_y_center = (v[:, :, 2:] - v[:, :, :-2]) / (2*dy)
        
        # Avg to X-faces (valid range)
        # u_y_center is defined for x=0..Nx+1. Face is between x and x+1.
        u_y_face = avg_x(u_y_center) # [Nx+1, Ny]
        v_y_face = avg_x(v_y_center)
        
        # Consistent shape for X-Flux calculation: [Nx+1, Ny]
        # u_x_face has Y ghost cells. Trim to center Ny.
        u_x_face = u_x_face[:, :, 1:-1] 
        v_x_face = v_x_face[:, :, 1:-1]
        dT_dx_face = dT_dx_face[:, :, 1:-1]
        
        div_u_face = u_x_face + v_y_face
        
        # Stresses at X-interfaces
        tau_xx_face = mu_visc * (2 * u_x_face - (2.0/3.0) * div_u_face)
        tau_xy_face = mu_visc * (u_y_face + v_x_face)
        
        # Heat and Work at X-interfaces
        u_face = avg_x(u[:, :, 1:-1]) #(u_i + u_i+1)/2
        v_face = avg_x(v[:, :, 1:-1])
        k_face = mu_visc * Cp / Pr
        
        E_flux_x = u_face * tau_xx_face + v_face * tau_xy_face + k_face * dT_dx_face
        
        # --- Flux Y (at j+1/2) ---
        # Same logic rotated
        
        # Y-derivatives at Y-interfaces (Nx+2, Ny+1)
        u_y_iface = diff_y(u)
        v_y_iface = diff_y(v)
        dT_dy_iface = diff_y(T)
        
        # X-derivatives at centers -> avg to Y-interfaces
        u_x_center = (u[:, 2:, :] - u[:, :-2, :]) / (2*dx) # [Nx, Ny+2]
        v_x_center = (v[:, 2:, :] - v[:, :-2, :]) / (2*dx)
        
        u_x_iface = avg_y(u_x_center) # [Nx, Ny+1]
        v_x_iface = avg_y(v_x_center)
        
        # Trim X ghosts of Y-diffs
        u_y_iface = u_y_iface[:, 1:-1, :]
        v_y_iface = v_y_iface[:, 1:-1, :]
        dT_dy_iface = dT_dy_iface[:, 1:-1, :]
        
        div_u_iface = u_x_iface + v_y_iface
        
        tau_yy_iface = mu_visc * (2 * v_y_iface - (2.0/3.0) * div_u_iface)
        tau_xy_iface = mu_visc * (u_y_iface + v_x_iface) # tau_yx = tau_xy
        
        u_iface = avg_y(u[:, 1:-1, :])
        v_iface = avg_y(v[:, 1:-1, :])
        k_iface = mu_visc * Cp / Pr
        
        E_flux_y = u_iface * tau_xy_iface + v_iface * tau_yy_iface + k_iface * dT_dy_iface
        
        # --- Compute Divergence ---
        # diff_x(FluxX) + diff_y(FluxY)
        # FluxX is [Nx+1, Ny]. diff_x -> [Nx, Ny]. Correct.
        
        # Momentum X: div(tau)_x = d(tau_xx)/dx + d(tau_yx)/dy
        # Use tau_xx from X-faces and tau_yx from Y-faces
        dtau_xx_dx = diff_x(tau_xx_face)
        dtau_yx_dy = diff_y(tau_xy_iface) # tau_xy at Y-face
        diff_mom_x = dtau_xx_dx + dtau_yx_dy
        
        # Momentum Y: div(tau)_y = d(tau_xy)/dx + d(tau_yy)/dy
        # Use tau_xy from X-faces and tau_yy from Y-faces
        dtau_xy_dx = diff_x(tau_xy_face)
        dtau_yy_dy = diff_y(tau_yy_iface)
        diff_mom_y = dtau_xy_dx + dtau_yy_dy
        
        # Energy: div(E_flux)
        diff_E = diff_x(E_flux_x) + diff_y(E_flux_y)
        
        return torch.stack([torch.zeros_like(diff_mom_x), diff_mom_x, diff_mom_y, diff_E], dim=1)

    def compute_dt(self, Q, cfl=0.5, mu=0.0):
        """
        Compute stable time step based on CFL condition.
        mu: dynamic viscosity (constant)
        """
        rho, u, v, p = self.get_primitive(Q)
        c = torch.sqrt(self.gamma * torch.abs(p) / (rho + 1e-8))
        
        # Convective spectral radius
        lambda_x = torch.abs(u) + c
        lambda_y = torch.abs(v) + c
        max_speed = torch.max(lambda_x.max(), lambda_y.max())
        
        # Convective dt
        dt_conv = self.dx / (max_speed + 1e-8)
        
        if mu > 0.0:
            # Diffusive dt limit (2D Stability)
            # dt < dx^2 / (4 * nu) for 2D explicit diffusion
            # nu = mu / rho.
            # Using rho_min is conservative.
            
            rho_min = rho.min()
            # If rho is too small/negative, clamp it for calc
            if rho_min < 1e-6: rho_min = 1e-6
            
            # Factor 4.0 for 2D Von Neumann stability
            dt_diff = (self.dx**2) * rho_min / (4.0 * mu + 1e-8)
            dt = cfl * min(dt_conv, dt_diff)
        else:
            dt = cfl * dt_conv
            
        return dt.item()

    @torch.no_grad()
    def step(self, Q, dt, mu=0.0):
        """RK2 Time stepping (gradient-free for memory efficiency)"""
        # First stage
        rhs1 = self.fluxes_explicit(Q) + self.viscous_fluxes_explicit(Q, mu)
        Q1 = Q + dt * rhs1
        
        # Second stage
        rhs2 = self.fluxes_explicit(Q1) + self.viscous_fluxes_explicit(Q1, mu)
        Q_next = 0.5 * Q + 0.5 * (Q1 + dt * rhs2)
        
        return Q_next

    def run(self, Q0, t_end, dt, mu=0.0):
        Q = Q0.clone()
        t = 0.0
        trajectory = [Q.cpu()]
        
        while t < t_end:
            # Simple adaptive dt based on CFL could be added here
            Q = self.step(Q, dt, mu)
            t += dt
            trajectory.append(Q.cpu())
            
        return torch.stack(trajectory)
