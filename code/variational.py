import numpy as np
import numpy.polynomial.polynomial as P
from scipy.optimize import minimize
from .rational import xi


class VariationalAnsatzApproach:
    """Variational ansatz approach for optimization."""

    NAME = "Variational Ansatz"

    def __init__(self, integrator):
        """
        Initialize with an integration manager.

        Args:
            integrator: IntegrationManager instance for numerical integration
        """
        self.integrator = integrator

    def run(self, model, error_type, orders, mu_val, optimizer, figs_dir, initial_guess_method='rational'):
        """
        Run the variational optimization.

        Args:
            model: Physical model instance
            error_type (str): Type of error ('energy' or 'fidelity')
            orders (list): List of (order_u, order_v) tuples
            mu_val (float): Chemical potential value
            optimizer (str): Optimization method
            figs_dir (str): Directory for saving figures
            initial_guess_method (str): Method for initial guess

        Returns:
            list: Results for each order pair
        """
        print(f"--- Running {self.NAME} for {model.NAME} (Optimizer: {optimizer}, Initial Guess: {initial_guess_method}) ---")
        objective_func = self._create_objective_function(model, error_type, mu_val)
        constraint_func = self._create_constraint_function(model, mu_val)
        results = []
        bz_volume = model.get_bz_volume()

        for order_u, order_v in orders:
            print(f"\n--- Optimizing {error_type} for orders u:{order_u}, v:{order_v} ---")
            constraints = [{'type': 'eq', 'fun': constraint_func, 'args': (order_u, order_v)}]

            n_guess = max(order_u, order_v)
            if initial_guess_method == 'rational':
                print("Generating initial guess from Rational Approximation...")
                initial_coeffs = self._get_rational_coeffs(model, mu_val, n_guess, order_u, order_v)
            else:
                initial_coeffs = np.random.rand(order_u + order_v + 2)
                norm_val = constraint_func(initial_coeffs, order_u, order_v) + bz_volume
                initial_coeffs /= np.sqrt(norm_val / bz_volume)

            print("Calculating cost of initial guess...")
            initial_cost_raw = objective_func(initial_coeffs, order_u, order_v)
            initial_cost_normalized = initial_cost_raw / bz_volume if bz_volume > 0 else initial_cost_raw
            print(f"Initial guess cost: {initial_cost_normalized:.8f}")

            print(f"Starting {optimizer} optimization...")

            iteration_counter = [0]

            def optimization_callback(xk):
                """
                Callback function for optimization iterations.
                Called at each iteration of the Powell algorithm.

                Args:
                    xk: Current parameter values (coefficients)
                """
                iteration_counter[0] += 1
                print("asdd")
                current_cost = objective_func(xk, order_u, order_v) / bz_volume
                print(f"  Iter: {iteration_counter[0]}, Current Cost: {current_cost:.6e}")

            # Define options for different optimizers
            optimizer_options = {
                'Nelder-Mead': {'disp': True, 'maxiter': 2000, 'adaptive': True, 'ftol': 1e-12},
                'L-BFGS-B': {'disp': True, 'maxiter': 500, 'gtol': 1e-9},
                'SLSQP': {'disp': True, 'maxiter': 500, 'ftol': 1e-9},
                'COBYLA': {'disp': True, 'maxiter': 2000, 'tol': 1e-15},
                'Powell': {'disp': True, 'maxiter': 2000, 'ftol': 1e-4}
            }
            options = optimizer_options.get(optimizer, {'disp': True})

            # Run optimization
            if optimizer == 'SLSQP':
                constraints = [{'type': 'eq', 'fun': constraint_func, 'args': (order_u, order_v)}]
                opt_result = minimize(
                    objective_func, initial_coeffs, args=(order_u, order_v),
                    method=optimizer, constraints=constraints, options=options
                )
            else:
                opt_result = minimize(
                    objective_func, initial_coeffs, args=(order_u, order_v),
                    method=optimizer, options=options,
                    callback=optimization_callback
                )

            final_coeffs = opt_result.x
            final_error = opt_result.fun / bz_volume if bz_volume > 0 else opt_result.fun
            print(final_error)

            result_data = {
                "orders": (order_u, order_v),
                "error_init": initial_cost_normalized,
                "error": final_error,
                "success": opt_result.success,
                "u_coeffs": final_coeffs[:order_u+1].tolist(),
                "v_coeffs": final_coeffs[order_u+1:].tolist(),
                "u_coeffs_init": final_coeffs[:order_u+1].tolist(),
                "v_coeffs_init": final_coeffs[order_u+1:].tolist(),
            }
            results.append(result_data)

            # Plot comparison for each completed order
            rational_coeffs = self._get_rational_coeffs(model, mu_val, n_guess, order_u, order_v)
            from .utils import plot_wavefunction_comparison
            plot_wavefunction_comparison(result_data, rational_coeffs, model, mu_val, error_type, figs_dir=figs_dir)

        return results

    def _get_rational_coeffs(self, model, mu, n, order_u, order_v):
        """
        Get rational approximation coefficients as initial guess.

        Args:
            model: Physical model instance
            mu (float): Chemical potential
            n (int): Order parameter
            order_u (int): Order of u polynomial
            order_v (int): Order of v polynomial

        Returns:
            np.array: Combined coefficients
        """
        if n <= 0:
            return np.random.rand(order_u + order_v + 2)

        k_vals = np.arange(n)
        xi_n_k = xi(n)**k_vals
        u_coeffs_raw = P.polyfromroots(-xi_n_k)
        v_coeffs_raw = P.polyfromroots(xi_n_k)

        def adjust_coeffs(coeffs, target_order):
            """Adjust coefficients to target order."""
            new_coeffs = np.zeros(target_order + 1)
            n_copy = min(len(coeffs), len(new_coeffs))
            new_coeffs[:n_copy] = coeffs[:n_copy]
            return new_coeffs

        u_coeffs = adjust_coeffs(u_coeffs_raw, order_u)
        v_coeffs = adjust_coeffs(v_coeffs_raw, order_v)
        initial_coeffs = np.concatenate([u_coeffs, v_coeffs])

        return initial_coeffs

    def _create_constraint_function(self, model, mu):
        """Create constraint function for normalization."""
        def constraint(coeffs, order_u, order_v):
            if model.DIM == 1:
                norm_integral = self.integrator.quad(
                    lambda k: self._norm_integrand(k, coeffs, order_u, order_v, model, mu),
                    *model.INTEGRATION_LIMITS
                )[0]
            elif model.DIM == 2:
                (kx_lim, ky_lim) = model.INTEGRATION_LIMITS
                norm_integral = self.integrator.dblquad(
                    lambda ky, kx: self._norm_integrand((kx, ky), coeffs, order_u, order_v, model, mu),
                    kx_lim[0], kx_lim[1], ky_lim[0], ky_lim[1]
                )[0]
            return norm_integral - model.get_bz_volume()
        return constraint

    def _norm_integrand(self, k, coeffs, order_u, order_v, model, mu):
        """Integrand for normalization constraint."""
        u_coeffs = coeffs[:order_u+1]
        v_coeffs = coeffs[order_u+1:]
        k_args = k if isinstance(k, tuple) else (k,)
        basis = model.basis(*k_args)
        pref = model.prefactor(*k_args)
        uk = P.polyval(basis, u_coeffs)
        vk = pref * P.polyval(basis, v_coeffs)
        return np.abs(uk)**2 + np.abs(vk)**2

    def _create_objective_function(self, model, error_type, mu_val):
        """Create objective function for optimization."""
        def objective(coeffs, order_u, order_v):
            total_integral = 0.0

            if model.DOMAIN_TYPE == '1D':
                total_integral = self.integrator.quad(
                    lambda k: self._error_integrand(k, coeffs, order_u, order_v, model, mu_val, error_type),
                    *model.INTEGRATION_LIMITS
                )[0]

            elif model.DOMAIN_TYPE in ['Rectangle', 'L-Shape', 'DiagonalStrip', 'DiagonalStrip2']:
                # 2D Cartesian
                def cartesian_integrand(ky, kx):
                    return self._error_integrand((kx, ky), coeffs, order_u, order_v, model, mu_val, error_type)

                for region in model.INTEGRATION_REGIONS:
                    (x_lims, y_min_func, y_max_func) = region
                    x_min, x_max = x_lims

                    integral_part, _ = self.integrator.dblquad(
                        cartesian_integrand,
                        x_min, x_max,
                        y_min_func, y_max_func
                    )
                    total_integral += integral_part

            elif model.DOMAIN_TYPE in ['Polar', 'AnnularSector']:
                # 2D Polar
                def polar_integrand(k_r, theta):
                    kx = k_r * np.cos(theta)
                    ky = k_r * np.sin(theta)
                    jacobian = k_r

                    cartesian_val = self._error_integrand((kx, ky), coeffs, order_u, order_v, model, mu_val, error_type)
                    return cartesian_val * jacobian

                for region in model.INTEGRATION_REGIONS:
                    (theta_lims, r_min_func, r_max_func) = region
                    theta_min, theta_max = theta_lims

                    integral_part, _ = self.integrator.dblquad(
                        polar_integrand,
                        theta_min, theta_max,
                        r_min_func, r_max_func
                    )
                    total_integral += integral_part

            return total_integral

        return objective

    def _error_integrand(self, k, coeffs, order_u, order_v, model, mu, error_type):
        """
        Calculate error integrand at given k-point.

        Args:
            k: k-point or tuple of k-values
            coeffs: Polynomial coefficients
            order_u (int): Order of u polynomial
            order_v (int): Order of v polynomial
            model: Physical model instance
            mu (float): Chemical potential
            error_type (str): Type of error ('energy' or 'fidelity')

        Returns:
            float: Error integrand value
        """
        u_coeffs = coeffs[:order_u+1]
        v_coeffs = coeffs[order_u+1:]  # Fixed: was coeffs[order_v+1:]
        k_args = k if isinstance(k, tuple) else (k,)
        basis = model.basis(*k_args)
        eps = model.epsilon(*k_args)
        pref = model.prefactor(*k_args)
        uk = P.polyval(basis, u_coeffs)
        vk = pref * P.polyval(basis, v_coeffs)
        norm_sq = np.abs(uk)**2 + np.abs(vk)**2

        if error_type == 'energy':
            if eps > 0:
                return eps * (np.abs(vk)**2 / norm_sq)
            else:
                return -eps * (np.abs(uk)**2 / norm_sq)

        elif error_type == 'fidelity':
            # Fidelity loss is -0.5 * log(P_correct_state)
            if eps > 0:  # Correct state is u=1, v=0. P_correct = |u|^2
                prob_u_sq = np.abs(uk)**2 / norm_sq
                return -0.5 * np.log(prob_u_sq)
            else:  # Correct state is u=0, v=1. P_correct = |v|^2
                prob_v_sq = np.abs(vk)**2 / norm_sq
                return -0.5 * np.log(prob_v_sq)

        else:
            raise ValueError(f"Unknown error_type: {error_type}")