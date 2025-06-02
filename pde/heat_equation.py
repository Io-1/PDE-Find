# Based on gray_scott.py structure citeturn0file0
"""
Heat equation PDE system implementation with configurable library build.
"""
import numpy as np
# from pysindy import PDELibrary

# Default parameter
ALPHA_DEFAULT: float = 0.01

def simulate_heat_equation(
    alpha: float = ALPHA_DEFAULT,
    N: int = 128,
    L: float = 1.0,
    nt: int = 128,
    dt: float = 0.1,
    initial_u: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1D heat equation (implicit diffusion) with Neumann (zero-flux) boundary conditions.

    Returns:
        U : u over time, shape (nt, N)
        dt : time step size
        x : spatial grid, shape (N,)
    """
    dx = L / (N - 1)
    x = np.linspace(0, L, N)

    # Default initial conditions: u=0 everywhere except central bump
    if initial_u is None:
        u = np.zeros(N)
        mid = N // 2
        width = max(1, N // 20)
        u[mid - width: mid + width] = 1.0
    else:
        u = initial_u.copy()

    # Precompute diffusion coefficient for implicit step
    alpha_coef = alpha * dt / dx**2

    # Tridiagonal coefficients
    a = -alpha_coef * np.ones(N)
    b = (1 + 2 * alpha_coef) * np.ones(N)
    c = -alpha_coef * np.ones(N)
    # Neumann BC adjustments
    c[0] = -2 * alpha_coef
    a[-2] = -2 * alpha_coef

    U = np.zeros((nt, N))

    for t in range(nt):
        # Diffusion (implicit via Thomas algorithm)
        u = thomas(a, b, c, u)
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
    alpha_param: float = ALPHA_DEFAULT,
) -> np.ndarray:
    """
    Build ground-truth coefficient matrix of shape (F,1) for the 1-D heat equation.
    """
    F = len(feature_names)
    coefs = np.zeros((F, 1))          # note 2-D shape

    mapping = {
        'uₓₓ': alpha_param,         # coefficient on u_xx
    }
    for i, name in enumerate(feature_names):
        if name in mapping:
            coefs[i, 0] = mapping[name]

    return coefs


class HeatEquation:
    def __init__(
        self,
        alpha: float = ALPHA_DEFAULT,
        N_space: int = 128,
        L_space: float = 1.0,
        N_time: int = 128,
        dt: float = 0.1,
    ):
        self.alpha = alpha
        self.N_space = N_space
        self.L_space = L_space
        self.N_time = N_time
        self.dt = dt

    @property
    def simulation_params(self) -> dict:
        return {
            'alpha': self.alpha,
            'N_space': self.N_space,
            'L_space': self.L_space,
            'N_time': self.N_time,
            'dt': self.dt,
        }

    def simulate(self) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Run the heat equation simulation and return data shaped for SINDy.

        Returns:
          data: ndarray of shape (n_space, n_time, 1)
          dt  : float
          x   : spatial grid (1D)
        """
        ics = self.initial_conditions()
        U, dt, x = simulate_heat_equation(
            alpha=self.alpha,
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
        # initial Gaussian distribution centered in domain
        sigma = self.L_space * 0.075
        center = self.L_space / 2
        gauss = np.exp(-(x - center)**2 / (2 * sigma**2))
        return {'u0': gauss}

    def boundary_conditions(self) -> dict:
        return {'type': 'Neumann (zero-flux)', 'description': 'du/dx=0'}

    def get_target_coefs(self, feature_names: list[str]) -> np.ndarray:
        from .heat_equation import get_target_coefs
        return get_target_coefs(feature_names, self.alpha)
