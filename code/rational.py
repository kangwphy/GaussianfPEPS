import numpy as np
import numpy.polynomial.polynomial as P


def xi(n):
    """
    Calculate ξ(n) = exp(-1/√n).

    Args:
        n (int): Order parameter

    Returns:
        float: ξ(n) value
    """
    if n <= 0:
        return np.inf
    return np.exp(-1.0 / np.sqrt(n))


def p(x, n):
    """
    Computes the product series Π(x + Ξ(n)^k) from k=0 to n-1.
    This version robustly handles both scalar and array inputs for x.

    Args:
        x: Input value(s) - can be scalar or array
        n (int): Order parameter

    Returns:
        Product series value(s)
    """
    if n <= 0:
        return 1.0

    x = np.asanyarray(x)

    k = np.arange(n)
    xi_n_k = xi(n)**k

    if x.ndim == 0:  # if scalar
        terms = x + xi_n_k
        return np.prod(terms)
    else:  # if array
        terms = x[..., np.newaxis] + xi_n_k
        return np.prod(terms, axis=1)


def g(x, n):
    """
    Calculates g(x, n) = p(-x, n) / p(x, n) using a direct product form
    from Newman's paper for better numerical stability. This computes
    the product of (ξ^k - x) / (ξ^k + x) directly.

    Args:
        x: Input value(s) - can be scalar or array
        n (int): Order parameter

    Returns:
        g(x, n) value(s)
    """
    x_arr = np.asanyarray(x)
    if n <= 0:
        return np.ones_like(x_arr)

    k_vals = np.arange(n)
    xi_n_k = xi(n)**k_vals

    if x_arr.ndim == 0:
        terms = (xi_n_k - x_arr) / (xi_n_k + x_arr)
        return np.prod(terms)
    else:
        numerator = xi_n_k - x_arr[..., np.newaxis]
        denominator = xi_n_k + x_arr[..., np.newaxis]
        terms = np.divide(numerator, denominator)
        return np.prod(terms, axis=1)


class RationalApproximationApproach:
    """Rational approximation approach for calculating errors."""

    NAME = "Rational Approximation"

    def __init__(self, integrator):
        """
        Initialize with an integration manager.

        Args:
            integrator: IntegrationManager instance for numerical integration
        """
        self.integrator = integrator

    def run(self, model, error_type, orders, mu_val):
        """
        Run the rational approximation calculation.

        Args:
            model: Physical model instance
            error_type (str): Type of error ('energy' or 'fidelity')
            orders (list): List of orders to calculate
            mu_val (float): Chemical potential value

        Returns:
            dict: Results containing orders and errors
        """
        print(f"--- Running {self.NAME} for {model.NAME} model ---")
        integrand_func = self._create_integrand(model, error_type)
        errors = []
        bz_volume = model.get_bz_volume()

        for n in orders:
            print(f"--- Processing order n = {n} ---")

            total_integral = 0.0

            # Multi-region integration logic
            if model.DOMAIN_TYPE == '1D':
                total_integral = self.integrator.quad(
                    integrand_func, *model.INTEGRATION_LIMITS, args=(n,)
                )[0]

            elif model.DOMAIN_TYPE in ['Rectangle', 'L-Shape', "DiagonalStrip", "DiagonalStrip2"]:
                # 2D Cartesian
                def cartesian_integrand(ky, kx, n_order):
                    return integrand_func(kx, ky, n_order)

                for region in model.INTEGRATION_REGIONS:
                    (x_lims, y_min_func, y_max_func) = region
                    x_min, x_max = x_lims

                    integral_part, _ = self.integrator.dblquad(
                        cartesian_integrand,
                        x_min, x_max,
                        y_min_func, y_max_func,
                        args=(n,)
                    )
                    total_integral += integral_part

            elif model.DOMAIN_TYPE in ['Polar', 'AnnularSector']:
                # 2D Polar
                def polar_integrand(k_r, theta, n_order):
                    kx = k_r * np.cos(theta)
                    ky = k_r * np.sin(theta)
                    jacobian = k_r

                    cartesian_val = integrand_func(kx, ky, n_order)
                    return cartesian_val * jacobian

                print("model.INTEGRATION_REGIONS", model.INTEGRATION_REGIONS)
                for region in model.INTEGRATION_REGIONS:
                    (theta_lims, r_min_func, r_max_func) = region
                    theta_min, theta_max = theta_lims

                    integral_part, _ = self.integrator.dblquad(
                        polar_integrand,
                        theta_min, theta_max,
                        r_min_func, r_max_func,
                        args=(n,)
                    )
                    print("integral_part = ", integral_part)
                    total_integral += integral_part

            error = total_integral / bz_volume if bz_volume > 0 else total_integral
            errors.append(error)
            print(f"Order {n}: Normalized Error = {error:.6e}")

        return {"orders": orders, "errors": errors}

    def _create_integrand(self, model, error_type):
        """Create the integrand function for error calculation."""
        def integrand(*args):
            n = args[-1]
            k_components = args[:-1]
            e = model.epsilon(*k_components)
            v_pref = np.abs(model.prefactor(*k_components))
            g_e, g_neg_e = g(e, n), g(-e, n)

            uv_ratio_sq = (min(np.abs(g_neg_e), np.abs(1.0 / g_e)) / (v_pref))**2
            vu_ratio_sq = (v_pref * min(np.abs(g_e), np.abs(1.0 / g_neg_e)))**2

            if error_type == 'energy':
                if e > 0:
                    return e / (1.0 + uv_ratio_sq)
                if e < 0:
                    return -e / (1.0 + vu_ratio_sq)
            elif error_type == 'fidelity':
                if e > 0:
                    return 0.5 * np.log(1.0 + vu_ratio_sq)
                if e < 0:
                    return 0.5 * np.log(1.0 + uv_ratio_sq)
            return 0.0
        return integrand

    def _create_integrand2(self, model, error_type):
        """Alternative integrand using p(x) functions."""
        def integrand(*args):
            n, mu = args[-1], args[-2]
            k_components = args[:-2]
            e = model.epsilon(*k_components)
            v_pref = np.abs(model.prefactor(*k_components))
            un = p(e, n)
            vn = v_pref * p(-e, n)

            if error_type == 'energy':
                if e > 0:
                    return e * (np.abs(vn)**2 / (np.abs(un)**2 + np.abs(vn)**2))
                if e < 0:
                    return -e * (np.abs(un)**2 / (np.abs(un)**2 + np.abs(vn)**2))
            elif error_type == 'fidelity':
                if e > 0:
                    return 0.5 * np.log(1.0+np.abs(vn)**2/np.abs(un)**2)
                if e < 0:
                    return 0.5 * np.log(1.0 + np.abs(un)**2/np.abs(vn)**2)
            return 0.0
        return integrand

    def _get_wavefunction_components(self, k_vals, mu_val, n, model):
        """
        Helper to get u(k), v(k), and their norms for plotting, using p(x).

        Args:
            k_vals: k-values for calculation
            mu_val: Chemical potential
            n (int): Order parameter
            model: Physical model instance

        Returns:
            tuple: (uk_vals, vk_vals, prob_u_sq, prob_v_sq)
        """
        e_vals = model.epsilon(k_vals)
        pref_vals = model.prefactor(k_vals)

        uk_vals = p(e_vals, n)
        vk_vals = pref_vals * p(-e_vals, n)

        norm_sq_vals = np.abs(uk_vals)**2 + np.abs(vk_vals)**2

        prob_u_sq = np.abs(uk_vals)**2 / norm_sq_vals
        prob_v_sq = np.abs(vk_vals)**2 / norm_sq_vals

        return uk_vals, vk_vals, prob_u_sq, prob_v_sq