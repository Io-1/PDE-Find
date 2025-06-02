"""
Gray-Scott PDE system implementation with configurable library build.
"""
import numpy as np
# from pysindy import PDELibrary


# Default parameters
DU_DEFAULT: float = 0.01
DV_DEFAULT: float = 0.005
F_DEFAULT:  float = 0.04
K_DEFAULT:  float = 0.06

def simulate_gray_scott(
    Du: float = 0.01,
    Dv: float = 0.005,
    F: float = 0.04,
    k: float = 0.06,
    N: int = 128,
    L: float = 1.0,
    nt: int = 128,
    dt: float = 0.1,
    initial_u: np.ndarray | None = None,
    initial_v: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1D Gray–Scott reaction-diffusion (explicit reactions, implicit diffusion)
    with Neumann (zero-flux) boundary conditions.
    
    Returns:
        U : u over time, shape (nt, N)
        V : v over time, shape (nt, N)
        dt : time step size
        x : spatial grid, shape (N,)
    """
    dx = L / (N - 1)
    x = np.linspace(0, L, N)

    # Default initial conditions: u=1 everywhere, v=0 except a small central perturbation
    if initial_u is None:
        u = np.ones(N)
    else:
        u = initial_u.copy()
    if initial_v is None:
        v = np.zeros(N)
        mid = N // 2
        width = max(1, N // 20)
        v[mid - width:mid + width] = 0.25
        u[mid - width:mid + width] = 0.75
    else:
        v = initial_v.copy()

    # Precompute diffusion coefficients for implicit step
    alpha_u = Du * dt / dx**2
    alpha_v = Dv * dt / dx**2

    # Tridiagonal coefficients for u
    a_u = -alpha_u * np.ones(N)
    b_u = (1 + 2 * alpha_u) * np.ones(N)
    c_u = -alpha_u * np.ones(N)
    # Neumann BC adjustments (zero-flux at boundaries)
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
        uv2 = u * v * v
        u_re = u + dt * (-uv2 + F * (1 - u))
        v_re = v + dt * ( uv2 - (F + k) * v)

        # Diffusion (implicit via Thomas)
        u = thomas(a_u, b_u, c_u, u_re)
        v = thomas(a_v, b_v, c_v, v_re)

        U[t] = u
        V[t] = v

    return U, V, dt, x

def thomas(a, b, c, d):
        """Solve tridiagonal system Ax = d with sub-diagonal a, diagonal b, super-diagonal c."""
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
    Du_param: float = DU_DEFAULT,
    Dv_param: float = DV_DEFAULT,
    F_param: float = F_DEFAULT,
    k_param: float = K_DEFAULT,
) -> np.ndarray:
    """
    Build ground-truth coefficient matrix matching feature names.
    """
    coefs = np.zeros((2, len(feature_names)))
    mapping1 = {'uvv': -1.0, 'u': -F_param, '1': F_param, 'uₓₓ': Du_param}
    mapping2 = {'uvv':  1.0, 'v': -(F_param + k_param),      'vₓₓ': Dv_param}
    for i, name in enumerate(feature_names):
        if name in mapping1:
            coefs[0, i] = mapping1[name]
        if name in mapping2:
            coefs[1, i] = mapping2[name]
    coefs = coefs.T
    return coefs


class GrayScott:
    def __init__(
        self,
        Du: float = 0.01,
        Dv: float = 0.005,
        F: float = 0.04,
        k: float = 0.06,
        N_space: int = 128,
        L_space: float = 1,
        N_time: int = 128,
        dt: float = 0.1,
    ):
        self.Du, self.Dv, self.F, self.k = Du, Dv, F, k
        self.N_space, self.L_space, self.N_time, self.dt = (
            N_space, L_space, N_time, dt
        )

    @property
    def simulation_params(self) -> dict:
        return {
            'Du': self.Du,
            'Dv': self.Dv,
            'F': self.F,
            'k': self.k,
            'N_space': self.N_space,
            'L_space': self.L_space,
            'N_time': self.N_time,
            'dt': self.dt,
        }

    def simulate(self) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Run the Gray–Scott simulation and return data shaped for SINDy.

        Returns:
          data: ndarray of shape (n_space, n_time, 2)
          dt  : float
          x   : spatial grid (1D)
        """
        ics = self.initial_conditions()
        U, V, dt, x = simulate_gray_scott(
            Du=self.Du, Dv=self.Dv,
            F=self.F,  k=self.k,
            N=self.N_space, L=self.L_space,
            nt=self.N_time, dt=self.dt,
            initial_u=ics['u0'], initial_v=ics['v0'],
        )
        uv = np.stack([U, V], axis=-1)
        data = uv.transpose(1, 0, 2)
        return data, dt, x

    def initial_conditions(self) -> dict:
        x = np.linspace(0, self.L_space, self.N_space)
        sigma = self.L_space * 0.075
        center = self.L_space / 2
        # gauss = (np.exp(-((x - center)**2) / (2*sigma**2))
        #          + 0.5 * np.exp(-((x - 0.7* self.L_space)**2)/(2*(sigma*1.5)**2)))
        gauss = (np.exp(- (x - 0.3*self.L_space)**2 / (2 * sigma**2))
         + 0.7 * np.exp(- (x - 0.7*self.L_space)**2 / (2 * (sigma*1.5)**2))) * 1

        u0 = np.ones(self.N_space) - gauss
        v0 = gauss
        return {'u0': u0, 'v0': v0}

    def boundary_conditions(self) -> dict:
        return {'type': 'Neumann (zero-flux)', 'description': 'du/dx=0, dv/dx=0'}

    def get_target_coefs(self, feature_names: list[str]) -> np.ndarray:
        from .gray_scott import get_target_coefs
        return get_target_coefs(feature_names, self.Du, self.Dv, self.F, self.k)

