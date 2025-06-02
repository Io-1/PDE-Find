# Based on gray_scott.py structure citeturn1file0
"""
Fisher-KPP PDE system implementation with configurable library build.
"""
import numpy as np
# from pysindy import PDELibrary


# Default parameters
D_DEFAULT: float = 0.01
R_DEFAULT: float = 1.0

def simulate_fisher_kpp(
    D: float = D_DEFAULT,
    r: float = R_DEFAULT,
    N: int = 128,
    L: float = 1.0,
    nt: int = 128,
    dt: float = 0.1,
    initial_u: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    1D Fisher–KPP reaction-diffusion (explicit reaction, implicit diffusion)
    with Neumann (zero-flux) boundary conditions.

    Returns:
        U : u over time, shape (nt, N)
        dt : time step size
        x : spatial grid, shape (N,)
    """
    dx = L / (N - 1)
    x = np.linspace(0, L, N)

    # Default initial condition: Gaussian bump
    if initial_u is None:
        sigma = L * 0.075
        center = L / 2
        u = np.exp(-(x - center)**2 / (2 * sigma**2))
    else:
        u = initial_u.copy()

    # Precompute diffusion coefficient for implicit step
    alpha = D * dt / dx**2

    # Tridiagonal coefficients
    a = -alpha * np.ones(N)
    b = (1 + 2 * alpha) * np.ones(N)
    c = -alpha * np.ones(N)
    # Neumann BC adjustments
    c[0] = -2 * alpha
    a[-2] = -2 * alpha

    U = np.zeros((nt, N))

    for t in range(nt):
        # Reaction (explicit)
        u_re = u + dt * (r * u * (1 - u))
        # Diffusion (implicit via Thomas algorithm)
        u = thomas(a, b, c, u_re)
        U[t] = u

    return U, dt, x

def thomas(a, b, c, d):
    """Solve tridiagonal system Ax = d with sub-diagonal a, diag b, super-diagonal c."""
    n = len(b)
    cp = np.empty(n - 1)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
    dp[-1] = (d[-1] - a[-2] * dp[-2]) / (b[-1] - a[-2] * cp[-2])
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x

def get_target_coefs(
    feature_names: list[str],
    D_param: float = D_DEFAULT,
    r_param: float = R_DEFAULT,
) -> np.ndarray:
    """
    Build ground‐truth coefficient matrix of shape (F,1) for Fisher–KPP,
    so it matches the multi‐species signature.
    """
    F = len(feature_names)
    coefs = np.zeros((F, 1))      # <-- note (F,1), not (F,)

    mapping = {
        'uₓₓ': D_param,    # u_xx
        'u':    r_param,    # u
        'uu': -r_param,    # u^2 → negative logistic term
    }
    for i, name in enumerate(feature_names):
        if name in mapping:
            coefs[i, 0] = mapping[name]

    return coefs

class FisherKPP:
    def __init__(
        self,
        D: float = D_DEFAULT,
        r: float = R_DEFAULT,
        N_space: int = 128,
        L_space: float = 1.0,
        N_time: int = 128,
        dt: float = 0.1,
    ):
        self.D = D
        self.r = r
        self.N_space = N_space
        self.L_space = L_space
        self.N_time = N_time
        self.dt = dt

    @property
    def simulation_params(self) -> dict:
        return {
            'D':       self.D,
            'r':       self.r,
            'N_space': self.N_space,
            'L_space': self.L_space,
            'N_time':  self.N_time,
            'dt':      self.dt,
        }

    def simulate(self) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Run the Fisher–KPP simulation and return data shaped for SINDy.

        Returns:
          data: ndarray of shape (n_space, n_time, 1)
          dt  : float
          x   : spatial grid (1D)
        """
        ics = self.initial_conditions()
        U, dt, x = simulate_fisher_kpp(
            D=self.D,
            r=self.r,
            N=self.N_space,
            L=self.L_space,
            nt=self.N_time,
            dt=self.dt,
            initial_u=ics['u0'],
        )
        u_data = U[..., np.newaxis]
        data = u_data.transpose(1, 0, 2)
        return data, dt, x

    def initial_conditions(self) -> dict:
        x = np.linspace(0, self.L_space, self.N_space)
        sigma = self.L_space * 0.075
        center = self.L_space / 2
        gauss = np.exp(-(x - center)**2 / (2 * sigma**2))
        return {'u0': gauss}

    def boundary_conditions(self) -> dict:
        return {'type': 'Neumann (zero-flux)', 'description': 'du/dx=0'}

    def get_target_coefs(self, feature_names: list[str]) -> np.ndarray:
        from .fisher_kpp import get_target_coefs
        return get_target_coefs(feature_names, self.D, self.r)
