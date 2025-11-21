import numpy as np


class BaseModel:
    """Base class for physical models."""

    DIM = 0
    NAME = "Base Model"
    INTEGRATION_LIMITS = None

    # @staticmethod
    def epsilon(*k):
        """Energy dispersion relation."""
        raise NotImplementedError

    # @staticmethod
    def prefactor(*k):
        """Prefactor for the model."""
        raise NotImplementedError

    # @classmethod
    def basis(cls, *k):
        """Basis function for the model."""
        return cls.epsilon(*k)

    # @classmethod
    def get_bz_volume(cls):
        """Calculate the Brillouin zone volume."""
        if cls.DIM == 1:
            k_min, k_max = cls.INTEGRATION_LIMITS
            return np.pi
        elif cls.DIM == 2:
            (kx_lim, ky_lim) = cls.INTEGRATION_LIMITS
            return np.pi**2
        return 1


class Model1D(BaseModel):
    """1D Tight-Binding model with NN and NNN hopping."""

    DIM = 1
    NAME = "1D Tight-Binding"
    INTEGRATION_LIMITS = (0, np.pi)

    def __init__(self, t1=1.0, t2=0.0, mu=0.0):
        """
        Initializes the 1D model with NN (t1) and NNN (t2) terms.

        Args:
            t1: Nearest neighbor hopping amplitude
            t2: Next-nearest neighbor hopping amplitude
            mu: Chemical potential
        """
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.mu = mu
        self.DOMAIN_TYPE = '1D'

        # Calculate maximum absolute energy for normalization
        candidates = [-2.0*t1 - 2.0*t2 - mu, 2.0*t1 - 2.0*t2 - mu]
        if t2 != 0 and abs(t1) <= abs(4.0 * t2):
            candidates.append(t1**2 / (4.0 * t2) + 2.0 * t2 - mu)

        self.absEmax = np.max(np.abs(np.array(candidates)))
        print("self.absEmax", self.absEmax)

    def epsilon(self, k):
        """Energy dispersion for 1D model."""
        return (- 2.0 * self.t1*np.cos(k) - 2.0 * self.t2*np.cos(2.0*k) - self.mu)/self.absEmax

    def prefactor(self, k):
        """Prefactor for 1D model."""
        return 2.0*(self.t1*np.sin(k) + 2.0*self.t2*np.sin(2.0*k))/self.absEmax


class Model2D(BaseModel):
    """2D Square Lattice model with NN, NNN, and next-next-nearest neighbor hopping."""

    DIM = 2
    NAME = "2D Square Lattice"
    INTEGRATION_LIMITS = ((0, np.pi), (0, np.pi))

    def __init__(self, t1=1.0, t2=0.0, t3=0.0, mu=0.0):
        """
        Initializes the 2D model with various hopping terms.

        Args:
            t1: Nearest neighbor hopping amplitude
            t2: Next-nearest neighbor hopping amplitude
            t3: Next-next-nearest neighbor hopping amplitude
            mu: Chemical potential
        """
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.mu = mu
        self.DOMAIN_TYPE = 'Rectangle'

        # Calculate energy extrema for normalization
        candidates_E = [
            -4.0*t1 - 4.0*t2 - 4.0*t3,  # (0, 0)
             4.0*t1 - 4.0*t2 - 4.0*t3,  # (pi, pi)
             4.0*t2 - 4.0*t3            # (pi, 0) or (0, pi)
        ]

        # Diagonal Inner Point (kx = ky)
        # condition: cos(k) = -t1 / (2t2 + 4t3)
        denom_diag = 2.0 * t2 + 4.0 * t3
        if abs(denom_diag) > 1e-9:
            u = -t1 / denom_diag
            if abs(u) <= 1.0:
                val = -4.0*t1*u - (4.0*t2 + 8.0*t3)*u**2 + 4.0*t3
                candidates_E.append(val)

        # Axis Inner Points (e.g. kx=0, vary ky)
        if abs(t3) > 1e-9:
            # Case kx = 0 (cos=1)
            v1 = -(t1 + 2.0*t2) / (4.0*t3)
            if abs(v1) <= 1.0:
                val = -2.0*t1*(1.0+v1) - 4.0*t2*v1 - 2.0*t3*(1.0 + 2.0*v1**2 - 1.0)
                candidates_E.append(val)

            # Case kx = pi (cos=-1)
            v2 = -(t1 - 2.0*t2) / (4.0*t3)
            if abs(v2) <= 1.0:
                val = -2.0*t1*(-1.0+v2) + 4.0*t2*v2 - 4.0*t3*v2**2
                candidates_E.append(val)

        e_extrema = np.array(candidates_E) - mu
        self.absEmax = np.max(np.abs(e_extrema))
        print(f"t1:{t1}, t2:{t2}, t3:{t3}, mu:{mu} => |E|_max = {self.absEmax}")

    def epsilon(self, kx, ky):
        """Energy dispersion for 2D model."""
        return (- 2.0 * self.t1 * (np.cos(kx) + np.cos(ky)) -
                4.0 * self.t2 * (np.cos(kx) * np.cos(ky)) -
                2.0 * self.t3 * (np.cos(2*kx) + np.cos(2*ky)) -
                self.mu)/self.absEmax

    def prefactor(self, kx, ky):
        """Prefactor for 2D model."""
        vx = 2.0 * self.t1*np.sin(kx) + 4.0 * self.t2 * np.sin(kx) * np.cos(ky) + 4.0 * self.t3 * np.sin(2*kx)
        vy = 2.0 * self.t1*np.sin(ky) + 4.0 * self.t2 * np.sin(ky) * np.cos(kx) + 4.0 * self.t3 * np.sin(2*ky)
        return (vx + 1j * vy) / self.absEmax