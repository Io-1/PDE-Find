# Based on gray_scott.py structure citeturn2file0
"""
Brusselator reaction-diffusion PDE system implementation with configurable library build.
"""
import numpy as np
# from pysindy import PDELibrary

# Default parameters
A_DEFAULT: float = 1.0
B_DEFAULT: float = 3.0
Du_DEFAULT: float = 0.01
Dv_DEFAULT: float = 0.005

def simulate_brusselator(
    A: float = A_DEFAULT,
    B: float = B_DEFAULT,
    Du: float = Du_DEFAULT,
    Dv: float = Dv_DEFAULT,
    N: int = 128,
    L: float = 1.0,
    nt: int = 128,
    dt: float = 0.1,
    initial_u: np.ndarray | None = None,
    initial_v: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    1D Brusselator reaction-diffusion (explicit reactions, implicit diffusion)
    with Neumann (zero-flux) boundary conditions.

    u_t = A - (B+1)*u + u^2*v + Du*u_xx
    v_t = B*u - u^2*v + Dv*v_xx

    Returns:
        U : u over time, shape (nt, N)
        V : v over time, shape (nt, N)
        dt: time step size
        x : spatial grid, shape (N,)
    """
    dx = L / (N - 1)
    x = np.linspace(0, L, N)

    # Initial conditions: uniform plus small perturbation
    if initial_u is None:
        u = A * np.ones(N)
    else:
        u = initial_u.copy()
    if initial_v is None:
        v = B / A * np.ones(N)
        mid = N // 2
        width = max(1, N // 20)
        u[mid-width:mid+width] += 0.1
        v[mid-width:mid+width] -= 0.1
    else:
        v = initial_v.copy()

    # Precompute diffusion coefficients for implicit step
    alpha_u = Du * dt / dx**2
    alpha_v = Dv * dt / dx**2

    # Tridiagonal coefficients for u
    a_u = -alpha_u * np.ones(N)
    b_u = (1 + 2 * alpha_u) * np.ones(N)
    c_u = -alpha_u * np.ones(N)
    c_u[0] = -2 * alpha_u
    a_u[-2] = -2 * alpha_u

    # Tridiagonal coefficients for v
    a_v = -alpha_v * np.ones(N)
    b_v = (1 + 2 * alpha_v) * np.ones(N)
    c_v = -alpha_v * np.ones(N)
    c_v[0] = -2 * alpha_v
    a_v[-2] = -2 * alpha_v

    U = np.zeros((nt, N))
    V = np.zeros((nt, N))

    for t in range(nt):
        # Reaction (explicit)
        uv2 = u*u*v
        u_re = u + dt * (A - (B + 1)*u + uv2)
        v_re = v + dt * (B*u - uv2)

        # Diffusion (implicit via Thomas algorithm)
        u = thomas(a_u, b_u, c_u, u_re)
        v = thomas(a_v, b_v, c_v, v_re)

        U[t] = u
        V[t] = v

    return U, V, dt, x

def thomas(a, b, c, d):
    """Solve tridiagonal system Ax = d with sub-diagonal a, diag b, super-diagonal c."""
    n = len(b)
    cp = np.empty(n-1)
    dp = np.empty(n)
    cp[0] = c[0]/b[0]
    dp[0] = d[0]/b[0]
    for i in range(1, n-1):
        denom = b[i] - a[i-1]*cp[i-1]
        cp[i] = c[i]/denom
        dp[i] = (d[i] - a[i-1]*dp[i-1])/denom
    dp[-1] = (d[-1] - a[-2]*dp[-2])/(b[-1] - a[-2]*cp[-2])
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x

def get_target_coefs(
    feature_names: list[str],
    A_param: float = A_DEFAULT,
    B_param: float = B_DEFAULT,
    Du_param: float = Du_DEFAULT,
    Dv_param: float = Dv_DEFAULT,
) -> np.ndarray:
    """
    Build ground-truth coefficient matrix for the Brusselator.
    """
    coefs = np.zeros((2, len(feature_names)))
    mapping1 = {
        '1':      A_param,
        'u':    -(B_param + 1),
        'uuv': 1.0,
        'uₓₓ':  Du_param,
    }
    mapping2 = {
        'u':     B_param,
        'uuv': -1.0,
        'uₓₓ':  Dv_param,
    }
    for i, name in enumerate(feature_names):
        if name in mapping1:
            coefs[0, i] = mapping1[name]
        if name in mapping2:
            coefs[1, i] = mapping2[name]
    return coefs.T

class Brusselator:
    def __init__(
        self,
        A: float = A_DEFAULT,
        B: float = B_DEFAULT,
        Du: float = Du_DEFAULT,
        Dv: float = Dv_DEFAULT,
        N_space: int = 128,
        L_space: float = 1.0,
        N_time: int = 128,
        dt: float = 0.1,
    ):
        self.A, self.B, self.Du, self.Dv = A, B, Du, Dv
        self.N_space, self.L_space, self.N_time, self.dt = (
            N_space, L_space, N_time, dt
        )

    @property
    def simulation_params(self) -> dict:
        return {
            'A':     self.A,
            'B':     self.B,
            'Du':    self.Du,
            'Dv':    self.Dv,
            'N_space': self.N_space,
            'L_space': self.L_space,
            'N_time':  self.N_time,
            'dt':      self.dt,
        }

    def simulate(self) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Run the Brusselator simulation and return data shaped for SINDy.

        Returns:
          data: ndarray of shape (n_space, n_time, 2)
          dt  : float
          x   : spatial grid (1D)
        """
        ics = self.initial_conditions()
        U, V, dt, x = simulate_brusselator(
            A=self.A,
            B=self.B,
            Du=self.Du,
            Dv=self.Dv,
            N=self.N_space,
            L=self.L_space,
            nt=self.N_time,
            dt=self.dt,
            initial_u=ics['u0'],
            initial_v=ics['v0'],
        )
        uv = np.stack([U, V], axis=-1)
        data = uv.transpose(1, 0, 2)
        return data, dt, x

    def initial_conditions(self) -> dict:
        x = np.linspace(0, self.L_space, self.N_space)
        # Base steady state
        u0 = self.A * np.ones(self.N_space)
        v0 = (self.B / self.A) * np.ones(self.N_space)

        # Add a small random perturbation:
        # – seed for reproducibility (optional)
        np.random.seed(42)
        noise_amp = 0.01  # 1% noise
        # u0 += noise_amp * np.random.randn(self.N_space)
        # v0 += noise_amp * np.random.randn(self.N_space)

        # (Or instead of random, you could use a sinusoidal bump:)
        u0 += 0.01 * np.sin(2*np.pi * x / self.L_space)
        v0 += 0.01 * np.cos(2*np.pi * x / self.L_space)

        return {'u0': u0, 'v0': v0}

    def boundary_conditions(self) -> dict:
        return {'type': 'Neumann (zero-flux)', 'description': 'du/dx=0, dv/dx=0'}

    def get_target_coefs(self, feature_names: list[str]) -> np.ndarray:
        from .brusselator import get_target_coefs
        return get_target_coefs(feature_names, self.A, self.B, self.Du, self.Dv)
