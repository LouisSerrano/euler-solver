
import torch
import numpy as np

def analyze_resolution(N=128, L=2*np.pi, nu=1e-4, U_scale=1.0):
    """
    Analyze if the grid resolution N is sufficient for Direct Numerical Simulation (DNS)
    at the given viscosity nu.
    """
    print(f"\n--- Resolution Analysis for N={N}, ν={nu}, L={L:.2f} ---")
    
    # 1. Estimate Reynolds Number
    # Re = U * L / nu
    # Taking L as the integral scale (domain size)
    Re = U_scale * L / nu
    print(f"Global Reynolds Number (approx): Re ≈ {Re:.1e}")
    
    # 2. Estimate Kolmogorov Length Scale (eta)
    # epsilon ~ U^3 / L (Energy dissipation rate)
    # eta = (nu^3 / epsilon)^(1/4)
    epsilon_est = (U_scale**3) / L
    eta = (nu**3 / epsilon_est)**(1/4)
    print(f"Estimated Kolmogorov scale (η): {eta:.6f}")
    
    # 3. Grid Statistics
    # dx = L / N
    dx = L / N
    print(f"Grid spacing (dx): {dx:.6f}")
    
    # 4. Effective Resolution (Dealiasing)
    # With 2/3 rule, max effective k is N/3
    k_max_eff = N / 3
    dx_effective = np.pi / k_max_eff # Smallest resolved half-wave
    print(f"Effective grid spacing (2/3 rule): {dx_effective:.6f}")
    
    # 5. Resolution Ratio
    ratio = dx / eta
    print(f"Resolution Ratio (dx / η): {ratio:.2f}")
    
    # 6. Conclusion
    print("\n--- Verdict ---")
    if ratio < 2.0:
        print("✅ well-resolved DNS. Viscous physics should be accurate.")
    elif ratio < 5.0:
        print("⚠️  Marginally resolved. Smallest scales might be under-damped.")
    else:
        print("❌ UNDER-RESOLVED. The grid is too coarse to see physical viscosity.")
        print(f"   The numerical truncation (dealiasing) acts as the dominant dissipation.")
        print(f"   At this resolution, NS(ν={nu}) will look identical to Euler.")
        
        # Calculate required N
        # Want dx ~ eta => L/N ~ eta => N ~ L/eta
        N_required = int(L / eta)
        print(f"   -> Required N for DNS: ~{N_required}")

if __name__ == "__main__":
    print("ANALYZING USER SCENARIOS:")
    analyze_resolution(N=128, nu=1e-3)
    analyze_resolution(N=128, nu=1e-4)
    analyze_resolution(N=256, nu=1e-4) # What if we double resolution?
