import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize
import numpy.polynomial.polynomial as P
import pandas as pd
import json
import os
# --- Set a nice style for the plots ---
plt.style.use('seaborn-v0_8-whitegrid')

# ==============================================================================
# PART 1: CORE RATIONAL APPROXIMATION FUNCTIONS (Unchanged)
# ==============================================================================
def xi(n):
    if n <= 0: return np.inf
    return np.exp(-1.0 / np.sqrt(n))

def p(x, n):
    """
    Computes the product series Π(x + Ξ(n)^k) from k=0 to n-1.
    This version robustly handles both scalar and array inputs for x.
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
    """
    x_arr = np.asanyarray(x)
    if n <= 0: return np.ones_like(x_arr)
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

# ==============================================================================
# PART 2: PHYSICAL MODELS (INTERFACE)
# ==============================================================================
class BaseModel:
    DIM = 0; NAME = "Base Model";INTEGRATION_LIMITS = None 
    # @staticmethod
    def epsilon(self,*k): raise NotImplementedError
    # @staticmethod
    def prefactor(self,*k): raise NotImplementedError
    # @classmethod
    def basis(self, *k): return self.epsilon(*k)
    @classmethod
    def get_bz_volume(cls):
        if cls.DIM == 1:
            k_min, k_max = cls.INTEGRATION_LIMITS
            return k_max - k_min
            # return 2*np.pi
        elif cls.DIM == 2:
            (kx_lim, ky_lim) = cls.INTEGRATION_LIMITS
            return (kx_lim[1] - kx_lim[0]) * (ky_lim[1] - ky_lim[0])
            # return (2*np.pi)**2
        return 1

class Model1D(BaseModel):
    DIM = 1; NAME = "1D Tight-Binding"; INTEGRATION_LIMITS = (0, np.pi)
    def __init__(self, t1 = 1.0, t2 = 0.0, mu = 0.0):
        """
        Initializes the 1D model with NN (t1) and NNN (t2) terms.
        t: hopping
        Defaults: t1=1.0
        """
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.mu = mu
        candidates = [-2.0*t1 - 2.0*t2 - mu, 2.0*t1 - 2.0*t2 - mu]
        if t2 != 0 and abs(t1) <= abs(4.0 * t2):
            candidates.append(t1**2 / (4.0 * t2) + 2.0 * t2-mu)
        self.absEmax = np.max(np.abs(np.array(candidates)))

    def epsilon(self, k): return (- 2.0 * self.t1*np.cos(k) - 2.0 * self.t2*np.cos(2.0*k) - self.mu)/self.absEmax
    def prefactor(self,k): return 2.0*(self.t1*np.sin(k) + 2.0*self.t2*np.sin(2.0*k))/self.absEmax

class Model2D(BaseModel):
    DIM = 2; NAME = "2D Square Lattice"; INTEGRATION_LIMITS = ((0, np.pi), (0, np.pi))

    def __init__(self, t1 = 1.0, t2 = 0.0, mu = 0.0):
        """
        Initializes the 1D model with NN (t1) and NNN (t2) terms.
        t: hopping
        Defaults: t1=1.0
        """
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.mu = mu

        candidates_E = [
            -2.0*t1*(1+1) - 4.0*t2*(1*1),    # E_raw(0, 0)   = -4*t1 - 4*t2
            -2.0*t1*(-1+1) - 4.0*t2*(-1*1),  # E_raw(pi, 0)  = 4*t2
            -2.0*t1*(1-1) - 4.0*t2*(1*-1),    # E_raw(0, pi)  = 4*t2 (相同)
            -2.0*t1*(-1-1) - 4.0*t2*(-1*-1)  # E_raw(pi, pi) = 4*t1 - 4*t2
        ]
        
        # Check Inner Point: dE/dkx = dE/dky = 0
        # Solution cos(kx) = cos(ky) = -t1/t2
        if t2 != 0 and abs(t1) <= abs(2.0 * t2):
            candidates_E.append(t1**2 / t2)

        e_extrema = np.array(list(set(candidates_E))) - mu
        self.absEmax = np.max(np.abs(e_extrema))
        if np.isclose(self.absEmax, 0): self.absEmax = 1.0
        print(f"t1:{t1}, t2:{t2}, mu:{mu} => |E|_max = {self.absEmax}")
    def epsilon(self, kx, ky): return (- 2.0 * self.t1 * (np.cos(kx) + np.cos(ky)) - 4.0 * self.t2 * (np.cos(kx) * np.cos(ky)) - self.mu)/self.absEmax

    def prefactor(self, kx, ky): 
        vx = 2.0 * self.t1*np.sin(kx) + 4.0 * self.t2 * np.sin(kx) * np.cos(ky)
        vy = 2.0 * self.t1*np.sin(ky) + 4.0 * self.t2 * np.sin(ky) * np.cos(kx)
        return 2.0*(vx + 1j * vy) / self.absEmax

# ==============================================================================
# PART 3: CALCULATION APPROACHES (Modified)
# ==============================================================================
class IntegrationManager:
    """A wrapper to manage global precision settings for integration."""
    def __init__(self, epsabs=1e-16, epsrel=1e-16):
        self.epsabs = epsabs
        self.epsrel = epsrel

    def quad(self, func, a, b, args=(), **kwargs):
        """Wrapper for scipy.integrate.quad with pre-set precision."""
        return quad(func, a, b, args=args, epsabs=self.epsabs, epsrel=self.epsrel, **kwargs)

    def dblquad(self, func, a, b, gfun, hfun, args=(), **kwargs):
        """Wrapper for scipy.integrate.dblquad with pre-set precision."""
        return dblquad(func, a, b, gfun, hfun, args=args, epsabs=self.epsabs, epsrel=self.epsrel, **kwargs)

class RationalApproximationApproach:
    NAME = "Rational Approximation"
    def __init__(self, integrator):
        self.integrator = integrator

    def run(self, model, error_type, orders, mu_val):
        print(f"--- Running {self.NAME} for {model.NAME} model ---")
        integrand_func = self._create_integrand(model, error_type)
        errors = []
        bz_volume = model.get_bz_volume()
        for n in orders:
            print(f"--- Processing order n = {n} ---")
            if model.DIM == 1:
                result, _ = self.integrator.quad(integrand_func, *model.INTEGRATION_LIMITS, args=(mu_val, n))
            elif model.DIM == 2:
                (kx_lim, ky_lim) = model.INTEGRATION_LIMITS
                result, _ = self.integrator.dblquad(lambda ky, kx: integrand_func(kx, ky, mu_val, n), kx_lim[0], kx_lim[1], ky_lim[0], ky_lim[1])
            error = result / bz_volume if bz_volume > 0 else result
            errors.append(error)
            print(f"Order {n}: Normalized Error = {error:.6e}")
        return {"orders": orders, "errors": errors}

    def _create_integrand(self, model, error_type):
        def integrand(*args):
            n, mu = args[-1], args[-2]
            k_components = args[:-2]
            e = model.epsilon(*k_components)
            v_pref = np.abs(model.prefactor(*k_components))
            g_e, g_neg_e = g(e, n), g(-e, n)
            uv_ratio_sq = (min(np.abs(g_neg_e), np.abs(1.0 / g_e)) / (v_pref))**2
            vu_ratio_sq = (v_pref * min(np.abs(g_e), np.abs(1.0 / g_neg_e)))**2
            if error_type == 'energy':
                if e > 0: return e / (1.0 + uv_ratio_sq)
                if e < 0: return -e / (1.0 + vu_ratio_sq)
            elif error_type == 'fidelity':
                if e > 0: return 0.5 * np.log(1.0 + vu_ratio_sq)
                if e < 0: return 0.5 * np.log(1.0 + uv_ratio_sq)
            return 0.0
        return integrand
    
    def _create_integrand2(self, model, error_type):
        def integrand(*args):
            n, mu = args[-1], args[-2]
            k_components = args[:-2]
            e = model.epsilon(*k_components)
            v_pref = np.abs(model.prefactor(*k_components))
            un = p(e, n); vn = v_pref * p(-e, n)
            if error_type == 'energy':
                if e > 0: return e * (np.abs(vn)**2 / (np.abs(un)**2 + np.abs(vn)**2))
                if e < 0: return -e * (np.abs(un)**2 / (np.abs(un)**2 + np.abs(vn)**2))
            elif error_type == 'fidelity':
                if e > 0: return 0.5 * np.log(1.0+np.abs(vn)**2/np.abs(un)**2)
                if e < 0: return 0.5 * np.log(1.0 + np.abs(un)**2/np.abs(vn)**2)
            return 0.0
        return integrand

    def _get_wavefunction_components(self, k_vals, mu_val, n, model):
        """Helper to get u(k), v(k), and their norms for plotting, using p(x)."""
        e_vals = model.epsilon(k_vals)
        pref_vals = model.prefactor(k_vals)

        uk_vals = p(e_vals, n)
        vk_vals = pref_vals * p(-e_vals, n)
        
        norm_sq_vals = np.abs(uk_vals)**2 + np.abs(vk_vals)**2
        # Guard against zero norm_sq for division
        # norm_sq_vals[norm_sq_vals < 1e-15] = 1e-15

        prob_u_sq = np.abs(uk_vals)**2 / norm_sq_vals
        prob_v_sq = np.abs(vk_vals)**2 / norm_sq_vals
        
        return uk_vals, vk_vals, prob_u_sq, prob_v_sq

class VariationalAnsatzApproach:
    NAME = "Variational Ansatz"
    def __init__(self, integrator):
        self.integrator = integrator

    def run(self, model, error_type, orders, mu_val, optimizer, figs_dir, initial_guess_method='rational'):
        print(f"--- Running {self.NAME} for {model.NAME} (Optimizer: {optimizer}, Initial Guess: {initial_guess_method}) ---")
        # print(f"--- Running {self.NAME} for {model.NAME} (Initial Guess: {initial_guess_method}) ---")
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
                这个函数会在 Powell 算法的每一次迭代结束时被调用。
                xk 是当前的参数（系数）
                """
                iteration_counter[0] += 1
                # if iteration_counter[0] % 1 == 0: # 每 10 次迭代打印一次
                #     # (可选) 你甚至可以重新计算当前点的能量值
                print("asdd")
                current_cost = objective_func(xk, order_u, order_v) / bz_volume
                print(f"  Iter: {iteration_counter[0]}, Current Cost: {current_cost:.6e}")
                    # print(f"  Iter: {iteration_counter[0]}...")
            # Define options for different optimizers
            optimizer_options = {
                'Nelder-Mead': {'disp': True, 'maxiter': 2000, 'adaptive': True, 'ftol': 1e-12},
                'L-BFGS-B': {'disp': True, 'maxiter': 500, 'gtol': 1e-9},
                'SLSQP': {'disp': True, 'maxiter': 500, 'ftol': 1e-9},
                'COBYLA': {'disp': True, 'maxiter': 2000, 'tol': 1e-15},
                'Powell': {'disp': True, 'maxiter': 2000, 'ftol': 1e-9}
            }
            options = optimizer_options.get(optimizer, {'disp': True})

            # Note: For gradient-based methods like 'L-BFGS-B' and 'SLSQP', `minimize`
            # will numerically approximate the gradient if `jac` is not provided.
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
            # print("Starting LBFGS optimization...")
            # # opt_result = minimize(
            # #     objective_func, initial_coeffs, args=(order_u, order_v),
            # #     # method='SLSQP', constraints=constraints,
            # #     # method='L-BFGS-B',
            # #     method = 'Nelder-Mead',
            # #     options={'disp': True, 'maxiter': 500, 'ftol': 1e-16,'gtol': 1e-9}
            # # )
            # opt_result = minimize(
            #     objective_func, initial_coeffs, args=(order_u, order_v),
            #     method = 'Nelder-Mead',
            #     # options={'disp': True, 'maxiter': 1000}
            #     options={
            #           'disp': True, 
            #           'maxiter': 1000, # 可以适当增加最大迭代次数
            #           'adaptive': True # 让算法对高维问题更鲁棒
            #       }
            # )#### Nelder-Mead, COBYLA, Powell。
            
            final_coeffs = opt_result.x
            # print(opt_result.fun)
            final_error = opt_result.fun / bz_volume if bz_volume > 0 else opt_result.fun
            print(final_error)
            result_data = {
                "orders": (order_u, order_v), "error_init": initial_cost_normalized, "error": final_error, "success": opt_result.success,
                "u_coeffs": final_coeffs[:order_u+1].tolist(),
                "v_coeffs": final_coeffs[order_u+1:].tolist(),
                "u_coeffs_init": final_coeffs[:order_u+1].tolist(),
                "v_coeffs_init": final_coeffs[order_u+1:].tolist(),
            }
            results.append(result_data)
            # Plot comparison for each completed order ---
            # Get the rational approximation coefficients of the same order for comparison
            rational_coeffs = self._get_rational_coeffs(model, mu_val, n_guess, order_u, order_v)
            plot_wavefunction_comparison(result_data, rational_coeffs, model, mu_val, error_type, figs_dir=figs_dir)

        return results

    def _get_rational_coeffs(self, model, mu, n, order_u, order_v):
        if n <= 0: return np.random.rand(order_u + order_v + 2)
        k_vals = np.arange(n)
        xi_n_k = xi(n)**k_vals
        u_coeffs_raw = P.polyfromroots(-xi_n_k)
        v_coeffs_raw = P.polyfromroots(xi_n_k)

        def adjust_coeffs(coeffs, target_order):
            new_coeffs = np.zeros(target_order + 1)
            n_copy = min(len(coeffs), len(new_coeffs))
            new_coeffs[:n_copy] = coeffs[:n_copy]
            return new_coeffs

        u_coeffs = adjust_coeffs(u_coeffs_raw, order_u)
        v_coeffs = adjust_coeffs(v_coeffs_raw, order_v)
        initial_coeffs = np.concatenate([u_coeffs, v_coeffs])
        
        bz_volume = model.get_bz_volume()
        # norm_val = self._create_constraint_function(model, mu)(initial_coeffs, order_u, order_v) + bz_volume
        # if norm_val < 1e-9: return np.random.rand(order_u + order_v + 2)
        
        # return initial_coeffs / np.sqrt(1 / bz_volume)
        return initial_coeffs

    def _create_constraint_function(self, model, mu):
        def constraint(coeffs, order_u, order_v):
            if model.DIM == 1:
                norm_integral = self.integrator.quad(lambda k: self._norm_integrand(k, coeffs, order_u, order_v, model, mu), *model.INTEGRATION_LIMITS)[0]
            elif model.DIM == 2:
                (kx_lim, ky_lim) = model.INTEGRATION_LIMITS
                norm_integral = self.integrator.dblquad(lambda ky, kx: self._norm_integrand((kx, ky), coeffs, order_u, order_v, model, mu), kx_lim[0], kx_lim[1], ky_lim[0], ky_lim[1])[0]
            return norm_integral - model.get_bz_volume()
        return constraint
    
    def _norm_integrand(self, k, coeffs, order_u, order_v, model, mu):
        u_coeffs = coeffs[:order_u+1]; v_coeffs = coeffs[order_u+1:]
        k_args = k if isinstance(k, tuple) else (k,)
        basis = model.basis(*k_args); pref = model.prefactor(*k_args)
        uk = P.polyval(basis, u_coeffs); vk = pref * P.polyval(basis, v_coeffs)
        return np.abs(uk)**2 + np.abs(vk)**2

    def _create_objective_function(self, model, error_type, mu_val):
        def objective(coeffs, order_u, order_v):
            if model.DIM == 1:
                return self.integrator.quad(lambda k: self._error_integrand(k, coeffs, order_u, order_v, model, mu_val, error_type), *model.INTEGRATION_LIMITS)[0]
            elif model.DIM == 2:
                (kx_lim, ky_lim) = model.INTEGRATION_LIMITS
                return self.integrator.dblquad(lambda ky, kx: self._error_integrand((kx, ky), coeffs, order_u, order_v, model, mu_val, error_type), kx_lim[0], kx_lim[1], ky_lim[0], ky_lim[1])[0]
        return objective

    def _error_integrand(self, k, coeffs, order_u, order_v, model, mu, error_type):
        u_coeffs = coeffs[:order_u+1]; v_coeffs = coeffs[order_v+1:]
        k_args = k if isinstance(k, tuple) else (k,)
        basis = model.basis(*k_args); eps = model.epsilon(*k_args); pref = model.prefactor(*k_args)
        uk = P.polyval(basis, u_coeffs); vk = pref * P.polyval(basis, v_coeffs)
        norm_sq = np.abs(uk)**2 + np.abs(vk)**2
        # if norm_sq < 1e-16: 
        #     print("k = ",k) 
        #     return 0.0

        if error_type == 'energy':
            if eps > 0: return eps * (np.abs(vk)**2 / norm_sq)
            else: return -eps * (np.abs(uk)**2 / norm_sq)
        
        elif error_type == 'fidelity':
            # Fidelity loss is -0.5 * log(P_correct_state)
            if eps > 0: # Correct state is u=1, v=0. P_correct = |u|^2
                prob_u_sq = np.abs(uk)**2 / norm_sq
                return -0.5 * np.log(prob_u_sq) 
            # if prob_u_sq > 1e-15 else -0.5 * np.log(1e-15)
            else: # Correct state is u=0, v=1. P_correct = |v|^2
                prob_v_sq = np.abs(vk)**2 / norm_sq
                return -0.5 * np.log(prob_v_sq) 
            # if prob_v_sq > 1e-15 else -0.5 * np.log(1e-15)
        
        else:
            raise ValueError(f"Unknown error_type: {error_type}")

# ==============================================================================
# PART 4: PLOTTING AND REPORTING
# ==============================================================================
# def save_results_to_txt(filename, data, header):
#     with open(filename, 'w') as f:
#         f.write(f"{header}\n=========================\n")
#         if "errors" in data:
#             for order, error in zip(data['orders'],data['init_error'], data['errors']):
#                 f.write(f"Order {order}: {error:.32f}\n")
#         else:
#             for res in data:
#                 f.write(f"Orders u={res['orders'][0]}, v={res['orders'][1]}: Error = {res['error']:.32f}\n")
#     print(f"Results successfully saved to '{filename}'")

# def save_results_to_dataframe(filename, data, header):
#     """
#     Saves results to a machine-friendly CSV file using Pandas.
#     Lists of coefficients are stored as JSON strings for easy retrieval.
#     """
#     if not data:
#         print("No data to save.")
#         return

#     print(f"Processing results for: {header}")
    
#     # --- Convert list of dictionaries to a DataFrame ---
#     df = pd.DataFrame(data)

#     # --- Pre-process the DataFrame for better usability ---
#     # Split the 'orders' tuple into separate 'order_u' and 'order_v' columns
#     if 'orders' in df.columns:
#         df[['order_u', 'order_v']] = pd.DataFrame(df['orders'].tolist(), index=df.index)
#         df = df.drop(columns=['orders']) # Drop the original tuple column

#     # Convert coefficient lists to JSON strings to store them properly in CSV
#     if 'u_coeffs' in df.columns:
#         df['u_coeffs'] = df['u_coeffs'].apply(json.dumps)
#     if 'v_coeffs' in df.columns:
#         df['v_coeffs'] = df['v_coeffs'].apply(json.dumps)

#     if 'u_coeffs_init' in df.columns:
#         df['u_coeffs_init'] = df['u_coeffs_init'].apply(json.dumps)
#     if 'v_coeffs_init' in df.columns:
#         df['v_coeffs_init'] = df['v_coeffs_init'].apply(json.dumps)

#     # Reorder columns for clarity
#     desired_order = [
#         'order_u', 'order_v', 'error_init', 'error', 'success', 
#         'u_coeffs_init','v_coeffs_init','u_coeffs', 'v_coeffs'
#     ]
#     # Keep only the columns that actually exist in the DataFrame
#     final_columns = [col for col in desired_order if col in df.columns]
#     df = df[final_columns]
    
#     # --- Save to CSV ---
#     df.to_csv(filename, index=False)
#     print(f"Results successfully saved to '{filename}' in CSV format.")

def save_results_to_txt(filename, data, header):
    """
    Saves human-readable results to a text file.
    FIXED: Correctly handles data from both Rational and Variational approaches.
    """
    with open(filename, 'w') as f:
        f.write(f"{header}\n=========================\n")
        
        # Check if data is from RationalApproach (a dict with 'errors')
        if isinstance(data, dict) and "errors" in data:
            # BUG FIX: Was zip(data['orders'], data['init_error'], data['errors'])
            # 'init_error' does not exist for RationalApproach
            for order, error in zip(data['orders'], data['errors']):
                f.write(f"Order {order}: {error:.32f}\n")
        
        # Check if data is from VariationalApproach (a list of dicts)
        elif isinstance(data, list):
            for res in data:
                f.write(f"Orders u={res['orders'][0]}, v={res['orders'][1]}: Error = {res['error']:.32f}\n")
        
        else:
            f.write("Data format not recognized.")
            
    print(f"Results successfully saved to '{filename}'")

def save_results_to_dataframe(filename, data, header):
    """
    Saves results to a machine-friendly CSV file using Pandas.
    FIXED: Correctly handles data from both Rational and Variational approaches.
    """
    if not data:
        print("No data to save.")
        return

    print(f"Processing results for: {header}")
    
    # --- Convert list/dict to a DataFrame ---
    df = pd.DataFrame(data)

    # --- Pre-process the DataFrame based on the approach ---
    
    # If it's Variational data (list of dicts), it will have 'error_init'
    if 'error_init' in df.columns:
        if 'orders' in df.columns:
            df[['order_u', 'order_v']] = pd.DataFrame(df['orders'].tolist(), index=df.index)
            df = df.drop(columns=['orders']) # Drop the original tuple column

        # Convert coefficient lists to JSON strings
        for col in ['u_coeffs', 'v_coeffs', 'u_coeffs_init', 'v_coeffs_init']:
            if col in df.columns:
                df[col] = df[col].apply(json.dumps)

        # Reorder columns for clarity
        desired_order = [
            'order_u', 'order_v', 'error_init', 'error', 'success', 
            'u_coeffs_init','v_coeffs_init','u_coeffs', 'v_coeffs'
        ]
        final_columns = [col for col in desired_order if col in df.columns]
        df = df[final_columns]
    
    # If it's Rational data (dict of lists), it will have 'errors'
    elif 'errors' in df.columns:
        # The DataFrame is already in the correct format:
        #   orders  errors
        # 0       1    0.10
        # 1       2    0.01
        # Just ensure the column order
        df = df[['orders', 'errors']]
        
    # --- Save to CSV ---
    df.to_csv(filename, index=False)
    print(f"Results successfully saved to '{filename}' in CSV format.")
def plot_rational_approx_error_scaling(data, model, error_type, mu_val, figs_dir='figs'):
    """Plots error scaling for the Rational Approximation approach."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    title = f'{model.NAME} {error_type.capitalize()} Error Scaling (Rational Approx., μ={mu_val})'
    fig.suptitle(title, fontsize=16)
    
    orders_np = np.array(data['orders'])
    errors_np = np.array(data['errors'])
    label = f"{error_type.capitalize()} Error"

    axes[0].plot(orders_np, errors_np, 'o-', label=label); axes[0].set_yscale('log'); axes[0].set_title('log(Error) vs. n'); axes[0].set_xlabel('Order (n)'); axes[0].set_ylabel('Error')
    axes[1].plot(np.sqrt(orders_np), errors_np, 's-', label=label); axes[1].set_yscale('log'); axes[1].set_title('log(Error) vs. sqrt(n)'); axes[1].set_xlabel('sqrt(n)')
    axes[2].plot(orders_np, errors_np, '^-', label=label); axes[2].set_xscale('log'); axes[2].set_yscale('log'); axes[2].set_title('log(Error) vs. log(n)'); axes[2].set_xlabel('Order (n)')
    
    for ax in axes: ax.grid(True, which="both", ls="--"); ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = os.path.join(figs_dir, f"{model.NAME.lower()}_rational_{error_type}_scaling_mu{mu_val}.png")
    plt.savefig(filename); plt.show()

def plot_variational_error_scaling(results, model, error_type, mu_val, figs_dir='figs'):
    """Plots error scaling for the Variational Ansatz approach."""
    if not results: return
    
    # Use the degree of the 'u' polynomial as the representative order for the x-axis
    orders = [res['orders'][0] for res in results]
    errors = [res['error'] for res in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    title = f'{model.NAME} {error_type.capitalize()} Error Scaling (Variational, μ={mu_val})'
    fig.suptitle(title, fontsize=16)

    orders_np = np.array(orders)
    errors_np = np.array(errors)
    label = f"{error_type.capitalize()} Error"

    # Plot 1: logy vs n
    axes[0].plot(orders_np, errors_np, 'o-', label=label)
    axes[0].set_yscale('log')
    axes[0].set_title('log(Error) vs. Polynomial Order')
    axes[0].set_xlabel('Order of u(k) polynomial')
    axes[0].set_ylabel('Error')

    # Plot 2: logy vs sqrt(n)
    axes[1].plot(np.sqrt(orders_np), errors_np, 's-', label=label)
    axes[1].set_yscale('log')
    axes[1].set_title('log(Error) vs. sqrt(Order)')
    axes[1].set_xlabel('sqrt(Order)')
    
    # Plot 3: log-log
    axes[2].plot(orders_np, errors_np, '^-', label=label)
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_title('log(Error) vs. log(Order)')
    axes[2].set_xlabel('Order of u(k) polynomial')

    for ax in axes:
        ax.grid(True, which="both", ls="--")
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = os.path.join(figs_dir, f"{model.NAME.lower()}_variational_{error_type}_error_scaling_mu{mu_val}.png")
    plt.savefig(filename); plt.show()

def plot_wavefunction_comparison(variational_result, rational_coeffs, model, mu_val, error_type, figs_dir='figs'):
    """
    Generates a 3x2 plot comparing the final optimized variational wavefunction
    against the rational approximation (initial guess).

    This function supports both 1D models and 2D models. For 2D models,
    it plots the data along the diagonal path from (0,0) to (pi,pi).
    """
    # --- 1. Extract Data and Coefficients ---
    order_u, order_v = variational_result['orders']
    final_error = variational_result['error']
    
    # Final optimized coefficients
    var_coeffs = np.concatenate([
        np.array(variational_result['u_coeffs']),
        np.array(variational_result['v_coeffs'])
    ])
    
    # Initial guess coefficients (from rational approximation)
    initial_coeffs = np.array(rational_coeffs, dtype=np.float64)
    
    num_points = 500

    # --- 2. Setup Momentum Path based on Model Dimension ---
    if model.DIM == 1:
        k_min, k_max = model.INTEGRATION_LIMITS
        k_vals = np.linspace(k_min, k_max, num_points)
        
        # Pre-calculate values for the 1D k-space
        basis_vals = model.basis(k_vals)
        pref_vals = model.prefactor(k_vals)
        eps_vals = model.epsilon(k_vals)
        
        # Set x-axis for plotting
        x_axis_data = k_vals
        x_axis_label = '$k$'
        
    elif model.DIM == 2:
        # Define a diagonal path from (0,0) to (pi,pi) in momentum space
        path_param = np.linspace(0, np.pi, num_points)
        kx_path = path_param
        ky_path = path_param
        
        # Pre-calculate values along the 2D path
        basis_vals = model.basis(kx_path, ky_path)
        pref_vals = model.prefactor(kx_path, ky_path)
        eps_vals = model.epsilon(kx_path, ky_path)
        
        # Set x-axis for plotting as the magnitude of the k-vector
        x_axis_data = np.sqrt(kx_path**2 + ky_path**2)
        x_axis_label = 'Momentum Path $|k|$ from $(0,0)$ to $(\pi,\pi)$'
    else:
        print(f"Wavefunction comparison plot is not implemented for DIM={model.DIM} models.")
        return

    # --- 3. Helper Function to Calculate Plotting Data ---
    def get_plotting_data(coeffs):
        """Calculates all necessary components for a given set of coefficients."""
        u_c = coeffs[:order_u + 1]
        v_c = coeffs[order_u + 1:]
        
        uk = P.polyval(basis_vals, u_c)
        vk = pref_vals * P.polyval(basis_vals, v_c)
        
        norm_sq = np.abs(uk)**2 + np.abs(vk)**2
        norm_sq[norm_sq < 1e-30] = 1e-30
        
        prob_u_sq = np.abs(uk)**2 / norm_sq
        prob_v_sq = np.abs(vk)**2 / norm_sq

        if error_type == 'energy':
            integrand = np.where(eps_vals > 0, 
                                 eps_vals * prob_v_sq, 
                                 -eps_vals * prob_u_sq)
        elif error_type == 'fidelity':
            prob_u_sq[prob_u_sq < 1e-30] = 1e-30
            prob_v_sq[prob_v_sq < 1e-30] = 1e-30
            integrand = np.where(eps_vals > 0, 
                                -0.5 * np.log(prob_u_sq), 
                                -0.5 * np.log(prob_v_sq))
        else:
            integrand = np.zeros_like(eps_vals)
            
        return uk, vk, prob_u_sq, prob_v_sq, integrand

    # --- 4. Calculate Data for Both Initial and Final States ---
    uk_init, vk_init, prob_u_init, prob_v_init, integrand_init = get_plotting_data(initial_coeffs)
    uk_final, vk_final, prob_u_final, prob_v_final, integrand_final = get_plotting_data(var_coeffs)

    # --- 5. Create the Figure and Subplots ---
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    title = (f'{model.NAME} Wavefunction Comparison (u:{order_u}, v:{order_v}, μ={mu_val})\n'
             f'Final Optimized {error_type.capitalize()} Error: {final_error:.6e}')
    fig.suptitle(title, fontsize=16)

    # Plot 1: Error Integrand
    axes[0, 0].plot(x_axis_data, integrand_init, 'r--', label='Initial Guess (Rational)', alpha=0.9)
    axes[0, 0].plot(x_axis_data, integrand_final, 'g-', label='Variational (Optimized)', lw=2)
    axes[0, 0].set_title(f'{error_type.capitalize()} Error Integrand')
    axes[0, 0].set_ylabel('Error Density')

    # Plot 2: Integrand Difference
    difference = integrand_final - integrand_init
    axes[0, 1].plot(x_axis_data, difference, 'k-')
    axes[0, 1].axhline(0, color='gray', lw=0.8, ls=':')
    axes[0, 1].set_title('Integrand Difference (Optimized - Initial)')
    axes[0, 1].set_ylabel('Difference')

    # Plot 3: Hole Probability |u|^2
    axes[1, 0].plot(x_axis_data, prob_u_init, 'r--', label='Initial Guess (Rational)', alpha=0.9)
    axes[1, 0].plot(x_axis_data, prob_u_final, 'g-', label='Variational (Optimized)', lw=2)
    axes[1, 0].set_title('Hole Probability $|u(k)|^2 / Z_k$')
    axes[1, 0].set_ylabel('Probability')

    # Plot 4: Occupation Probability |v|^2
    axes[1, 1].plot(x_axis_data, prob_v_init, 'r--', label='Initial Guess (Rational)', alpha=0.9)
    axes[1, 1].plot(x_axis_data, prob_v_final, 'g-', label='Variational (Optimized)', lw=2)
    axes[1, 1].set_title('Occupation Probability $|v(k)|^2 / Z_k$')
    axes[1, 1].set_ylabel('Probability')

    # Plot 5: u(k) component
    axes[2, 0].plot(x_axis_data, uk_init.real, 'r--', label='Initial Guess (Rational)', alpha=0.9)
    axes[2, 0].plot(x_axis_data, uk_final.real, 'g-', label='Variational (Optimized)', lw=2)
    axes[2, 0].set_title('Wavefunction Component $u(k)$')
    axes[2, 0].set_xlabel(x_axis_label); axes[2, 0].set_ylabel('Amplitude')

    # Plot 6: v(k) component
    axes[2, 1].plot(x_axis_data, vk_init.real, 'r--', label='Initial Guess (Rational)', alpha=0.9)
    axes[2, 1].plot(x_axis_data, vk_final.real, 'g-', label='Variational (Optimized)', lw=2)
    axes[2, 1].set_title('Wavefunction Component $v(k)$')
    axes[2, 1].set_xlabel(x_axis_label); axes[2, 1].set_ylabel('Amplitude')
    
    # --- 6. Final Touches and Saving ---
    # Add Fermi level line, ONLY for 1D models where it's a single point
    if model.DIM == 1 and abs(mu_val) <= 1.0:
        k_min, k_max = model.INTEGRATION_LIMITS
        k_f = np.arccos(-mu_val)
        if k_min <= k_f <= k_max:
            for ax in axes.flatten():
                ax.axvline(k_f, c='gray', ls=':', label='$k_F$')
    
    # Apply legends and grids to all subplots
    for ax in axes.flatten():
        ax.legend()
        ax.grid(True, which="both", ls="--")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    
    # savename = f"figs/{model.NAME.lower()}_variational_{error_type}_comparison_u{order_u}v{order_v}_mu{mu_val}.png"
    # plt.savefig(savename)
    savename = os.path.join(figs_dir, f"{model.NAME.lower()}_variational_{error_type}_comparison_u{order_u}v{order_v}_mu{mu_val}.png")
    plt.savefig(savename); print(f"Comparison plot saved to: {savename}"); plt.close(fig)
    print(f"Comparison plot saved to: {savename}")
    plt.close(fig)




# def plot_wavefunction_comparison(variational_result, rational_coeffs, model, mu_val, error_type):
#     """
#     Generates a 3x2 plot comparing the final optimized variational wavefunction
#     against the rational approximation (which serves as the initial guess).

#     This function is designed for 1D models.
#     """
#     if model.DIM != 1:
#         print("Wavefunction comparison plot is only implemented for 1D models.")
#         return

#     # --- 1. Extract Data and Coefficients ---
#     order_u, order_v = variational_result['orders']
#     final_error = variational_result['error']
    
#     # The final optimized coefficients from the variational result
#     var_coeffs = np.concatenate([
#         np.array(variational_result['u_coeffs']),
#         np.array(variational_result['v_coeffs'])
#     ])
    
#     # The rational_coeffs are used as the initial guess
#     initial_coeffs = np.array(rational_coeffs, dtype=np.float64)

#     # --- 2. Setup Plotting Grid and Common Variables ---
#     k_min, k_max = model.INTEGRATION_LIMITS
#     k_vals = np.linspace(k_min, k_max, 500)
    
#     # Pre-calculate basis, prefactor, and epsilon values for efficiency
#     basis_vals = model.basis(k_vals, mu=mu_val)
#     pref_vals = model.prefactor(k_vals)
#     eps_vals = model.epsilon(k_vals, mu=mu_val)

#     # --- 3. Helper Function to Calculate Plotting Data ---
#     def get_plotting_data(coeffs):
#         """Calculates all necessary components for a given set of coefficients."""
#         u_c = coeffs[:order_u + 1]
#         v_c = coeffs[order_u + 1:]
        
#         uk = P.polyval(basis_vals, u_c)
#         vk = pref_vals * P.polyval(basis_vals, v_c)
        
#         norm_sq = np.abs(uk)**2 + np.abs(vk)**2
#         # Add a small epsilon to prevent division by zero or log(0)
#         norm_sq[norm_sq < 1e-30] = 1e-30
        
#         prob_u_sq = np.abs(uk)**2 / norm_sq
#         prob_v_sq = np.abs(vk)**2 / norm_sq

#         # Calculate the error integrand based on the error type
#         if error_type == 'energy':
#             integrand = np.where(eps_vals > 0, 
#                                  eps_vals * prob_v_sq, 
#                                 -eps_vals * prob_u_sq)
#         elif error_type == 'fidelity':
#             # Add a floor to probabilities to avoid log(0)
#             prob_u_sq[prob_u_sq < 1e-30] = 1e-30
#             prob_v_sq[prob_v_sq < 1e-30] = 1e-30
#             integrand = np.where(eps_vals > 0, 
#                                 -0.5 * np.log(prob_u_sq), 
#                                 -0.5 * np.log(prob_v_sq))
#         else:
#             integrand = np.zeros_like(eps_vals)
            
#         return uk, vk, prob_u_sq, prob_v_sq, integrand

#     # --- 4. Calculate Data for Both Initial and Final States ---
#     uk_init, vk_init, prob_u_init, prob_v_init, integrand_init = get_plotting_data(initial_coeffs)
#     uk_final, vk_final, prob_u_final, prob_v_final, integrand_final = get_plotting_data(var_coeffs)

#     # --- 5. Create the Figure and Subplots ---
#     fig, axes = plt.subplots(3, 2, figsize=(16, 15))
#     title = (f'{model.NAME} Wavefunction Comparison (u:{order_u}, v:{order_v}, μ={mu_val})\n'
#              f'Final Optimized {error_type.capitalize()} Error: {final_error:.6e}')
#     fig.suptitle(title, fontsize=16)

#     # Plot 1: Error Integrand
#     axes[0, 0].plot(k_vals, integrand_init, 'r--', label='Initial Guess (Rational)', alpha=0.9)
#     axes[0, 0].plot(k_vals, integrand_final, 'g-', label='Variational (Optimized)', lw=2)
#     axes[0, 0].set_title(f'{error_type.capitalize()} Error Integrand')
#     axes[0, 0].set_ylabel('Error Density')

#     # Plot 2: Integrand Difference
#     difference = integrand_final - integrand_init
#     axes[0, 1].plot(k_vals, difference, 'k-')
#     axes[0, 1].axhline(0, color='gray', lw=0.8, ls=':')
#     axes[0, 1].set_title('Integrand Difference (Optimized - Initial)')
#     axes[0, 1].set_ylabel('Difference')

#     # Plot 3: Hole Probability |u|^2
#     axes[1, 0].plot(k_vals, prob_u_init, 'r--', label='Initial Guess (Rational)', alpha=0.9)
#     axes[1, 0].plot(k_vals, prob_u_final, 'g-', label='Variational (Optimized)', lw=2)
#     axes[1, 0].set_title('Hole Probability $|u(k)|^2 / Z_k$')
#     axes[1, 0].set_ylabel('Probability')

#     # Plot 4: Occupation Probability |v|^2
#     axes[1, 1].plot(k_vals, prob_v_init, 'r--', label='Initial Guess (Rational)', alpha=0.9)
#     axes[1, 1].plot(k_vals, prob_v_final, 'g-', label='Variational (Optimized)', lw=2)
#     axes[1, 1].set_title('Occupation Probability $|v(k)|^2 / Z_k$')
#     axes[1, 1].set_ylabel('Probability')

#     # Plot 5: u(k) component
#     axes[2, 0].plot(k_vals, uk_init.real, 'r--', label='Initial Guess (Rational)', alpha=0.9)
#     axes[2, 0].plot(k_vals, uk_final.real, 'g-', label='Variational (Optimized)', lw=2)
#     axes[2, 0].set_title('Wavefunction Component $u(k)$')
#     axes[2, 0].set_xlabel('$k$'); axes[2, 0].set_ylabel('Amplitude')

#     # Plot 6: v(k) component
#     axes[2, 1].plot(k_vals, vk_init.real, 'r--', label='Initial Guess (Rational)', alpha=0.9)
#     axes[2, 1].plot(k_vals, vk_final.real, 'g-', label='Variational (Optimized)', lw=2)
#     axes[2, 1].set_title('Wavefunction Component $v(k)$')
#     axes[2, 1].set_xlabel('$k$'); axes[2, 1].set_ylabel('Amplitude')
    
#     # --- 6. Final Touches and Saving ---
#     # Add Fermi level line if applicable
#     if abs(mu_val) <= 1.0:
#         k_f = np.arccos(-mu_val)
#         if k_min <= k_f <= k_max:
#             for ax in axes.flatten():
#                 ax.axvline(k_f, c='gray', ls=':', label='$k_F$')
    
#     # Apply legends and grids to all subplots
#     for ax in axes.flatten():
#         ax.legend()
#         ax.grid(True, which="both", ls="--")
        
#     plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    
#     # Save the figure
#     savename = f"figs/{model.NAME.lower()}_variational_{error_type}_comparison_u{order_u}v{order_v}_mu{mu_val}.png"
#     plt.savefig(savename)
#     print(f"Comparison plot saved to: {savename}")
#     plt.close(fig) # Close figure to prevent it from displaying in a loop

# def plot_wavefunction_comparison(variational_result, rational_coeffs, model, mu_val, error_type):
#     """
#     Generates a 2x2 plot comparing the final variational wavefunction
#     with the rational approximation guess for a single order.
#     """
#     if model.DIM != 1:
#         print("Wavefunction comparison plot is only implemented for 1D models.")
#         return

#     order_u, order_v = variational_result['orders']
#     var_u_coeffs = np.array(variational_result['u_coeffs'])
#     var_v_coeffs = np.array(variational_result['v_coeffs'])
    
#     rat_u_coeffs = rational_coeffs[:order_u+1]
#     rat_v_coeffs = rational_coeffs[order_u+1:]

#     title = f'{model.NAME} Comparison (u:{order_u},v:{order_v},μ={mu_val}), Final Error={variational_result["error"]:.6e}'
    
#     k_min, k_max = model.INTEGRATION_LIMITS
#     k_vals = np.linspace(k_min, k_max, 400)
#     basis_vals = model.basis(k_vals, mu=mu_val)
#     pref_vals = model.prefactor(k_vals)
    
#     # Calculate variational wavefunctions
#     uk_var = P.polyval(basis_vals, var_u_coeffs)
#     vk_var = pref_vals * P.polyval(basis_vals, var_v_coeffs)
#     norm_sq_var = np.abs(uk_var)**2 + np.abs(vk_var)**2
    
#     # Calculate rational approximation wavefunctions
#     uk_rat = P.polyval(basis_vals, rat_u_coeffs)
#     vk_rat = pref_vals * P.polyval(basis_vals, rat_v_coeffs)
#     norm_sq_rat = np.abs(uk_rat)**2 + np.abs(vk_rat)**2

#     fig, axes = plt.subplots(3, 2, figsize=(16, 12))
#     fig.suptitle(title, fontsize=16)

#     # Plot 1: u(k) and v(k) for both methods
#     axes[0, 0].plot(k_vals, uk_var, 'b-', label='u(k) Variational') 
#     axes[0, 0].plot(k_vals, uk_rat, 'b--', alpha=0.7, label='u(k) Rational')
#     # axes[0, 0].plot(k_vals, vk_var, 'r-', label='v(k) Variational')
#     # axes[0, 0].plot(k_vals, vk_rat, 'r--', alpha=0.7, label='v(k) Rational')
#     axes[0, 0].set_title('Wavefunction Components')
#     axes[0, 0].set_xlabel('$k$'); axes[0, 0].set_ylabel('Amplitude')

#     # Plot 2: For rational approximation for completeness
#     axes[0, 1].plot(k_vals, vk_var, 'r-', label='u(k) Variational')
#     axes[0, 1].plot(k_vals, vk_rat, 'r--', alpha=0.7, label='v(k) Rational')
#     axes[0, 1].set_title('Wavefunction Components')
#     axes[0, 1].set_xlabel('$k$'); axes[0, 1].set_ylabel('Amplitude')

#     # Plot 3: Hole probability |u|^2
#     axes[1, 0].plot(k_vals, np.abs(uk_var)**2 / norm_sq_var, 'b-', label='Variational')
#     axes[1, 0].plot(k_vals, np.abs(uk_rat)**2 / norm_sq_rat, 'b--', alpha=0.7, label='Rational')
#     axes[1, 0].set_title('Hole Probability $|u(k)|^2 / Z_k$')
#     axes[1, 0].set_xlabel('$k$'); axes[1, 0].set_ylabel('Probability')

#     # Plot 4: Occupation probability |v|^2
#     axes[1, 1].plot(k_vals, np.abs(vk_var)**2 / norm_sq_var, 'r-', label='Variational')
#     axes[1, 1].plot(k_vals, np.abs(vk_rat)**2 / norm_sq_rat, 'r--', alpha=0.7, label='Rational')
#     axes[1, 1].set_title('Occupation Probability $|v(k)|^2 / Z_k$')
#     axes[1, 1].set_xlabel('$k$'); axes[1, 1].set_ylabel('Probability')


#     order_u, order_v = variational_result['orders']
#     var_coeffs = np.concatenate([variational_result['u_coeffs'], variational_result['v_coeffs']])
    
#     title = f'{model.NAME} Comparison (u:{order_u},v:{order_v},μ={mu_val}), Final Error={variational_result["error"]:.6e}'
    
#     k_min, k_max = model.INTEGRATION_LIMITS
#     k_vals = np.linspace(k_min, k_max, 400)
#     basis_vals = model.basis(k_vals, mu=mu_val)
#     pref_vals = model.prefactor(k_vals)
#     eps_vals = model.epsilon(k_vals, mu=mu_val)
#     def get_plotting_data(coeffs):
#         u_c = coeffs[:order_u+1]; v_c = coeffs[order_u+1:]
#         uk = P.polyval(basis_vals, u_c); vk = pref_vals * P.polyval(basis_vals, v_c)
#         norm_sq = np.abs(uk)**2 + np.abs(vk)**2
#         # norm_sq[norm_sq < 1e-15] = 1.0
        
#         if error_type == 'energy':
#             integrand = np.where(eps_vals > 0, eps_vals * (np.abs(vk)**2 / norm_sq), -eps_vals * (np.abs(uk)**2 / norm_sq))
#         elif error_type == 'fidelity':
#             prob_u_sq = np.abs(uk)**2 / norm_sq
#             prob_v_sq = np.abs(vk)**2 / norm_sq
#             # prob_u_sq[prob_u_sq < 1e-15] = 1e-15
#             # prob_v_sq[prob_v_sq < 1e-15] = 1e-15
#             integrand = np.where(eps_vals > 0, -0.5 * np.log(prob_u_sq), -0.5 * np.log(prob_v_sq))
#         else:
#             integrand = np.zeros_like(eps_vals)
            
#         return uk, vk, norm_sq, integrand

#     uk_var, vk_var, norm_sq_var, integrand_var = get_plotting_data(var_coeffs)
#     uk_rat, vk_rat, norm_sq_rat, integrand_rat = get_plotting_data(rational_coeffs)

#     axes[2, 0].plot(k_vals, integrand_var, 'g-', label='Var Integrand')
#     axes[2, 0].plot(k_vals, integrand_rat, 'g--', alpha=0.7, label='Rat Integrand')
#     axes[2, 0].set_title(f'{error_type.capitalize()} Integrand'); axes[2, 0].set_xlabel('$k$'); axes[2, 0].set_ylabel('Error Density')

#     axes[2, 1].plot(k_vals, integrand_var - integrand_rat, 'k-', label='Var - Rat')
#     axes[2, 1].axhline(0, color='gray', lw=1, ls=':')
#     axes[2, 1].set_title(f'Integrand Difference, maximum {np.max(integrand_var - integrand_rat)}'); axes[2, 1].set_xlabel('$k$')
#     if abs(mu_val) <= 1.0:
#         k_f = np.arccos(-mu_val)
#         if k_min <= k_f <= k_max:
#             for ax in axes.flatten(): ax.axvline(k_f, c='gray', ls=':', label='$k_F$')
    
#     for ax in axes.flatten(): ax.legend(); ax.grid(True)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.savefig("figs/"+f"{model.NAME.lower()}_variational_{error_type}_comparison_u{order_u}v{order_v}_mu{mu_val}.png")
#     plt.close(fig) # Close the figure to avoid displaying it in interactive environments



# # ==============================================================================
# # PART 5: EXECUTION (Modified)
# # ==============================================================================
# if __name__ == '__main__':
#     # Define global precision settings
#     EPS_ABS = 1e-16
#     EPS_REL = 1e-16

#     # Create the integration manager instance
#     integrator_manager = IntegrationManager(epsabs=EPS_ABS, epsrel=EPS_REL)

#     # Now, pass the integrator manager to your classes
#     # APPROACH = RationalApproximationApproach(integrator_manager)
#     # or
#     APPROACH = VariationalAnsatzApproach(integrator_manager)
    
#     MODEL = Model1D()
#     # MODEL = Model2D()
#     # ERROR_TYPE = 'fidelity'
#     ERROR_TYPE = 'energy'
#     MU_VALUE = 0.0
    
#     if isinstance(APPROACH, RationalApproximationApproach):
#         ORDERS_TO_RUN = range(1, 100)
#     elif isinstance(APPROACH, VariationalAnsatzApproach):
#         if MODEL.DIM == 1:
#             # ORDERS_TO_RUN = [(4, 4), (6, 6), (8, 8)]
#             ORDERS_TO_RUN = [(n, n) for n in range(15,16)]
#         elif MODEL.DIM == 2:
#             ORDERS_TO_RUN = [(2, 2), (3, 3)]

#     # results_data = APPROACH.run(
#     #     model=MODEL, error_type=ERROR_TYPE, orders=ORDERS_TO_RUN, mu_val=MU_VALUE,
#     # )
#     results_data = APPROACH.run(
#         model=MODEL, error_type=ERROR_TYPE, orders=ORDERS_TO_RUN, mu_val=MU_VALUE,
#     )

#     header = f"{MODEL.NAME} {APPROACH.NAME} Results for {ERROR_TYPE}, mu={MU_VALUE}"
#     filename = f"{MODEL.NAME.lower()}_{APPROACH.NAME.replace(' ','')}_{ERROR_TYPE}_mu{MU_VALUE}.txt"
#     save_results_to_txt("data/"+filename, results_data, header)
    
#     if results_data:
#         if isinstance(APPROACH, RationalApproximationApproach):
#             plot_rational_approx_error_scaling(results_data, MODEL, ERROR_TYPE, MU_VALUE)
#         elif isinstance(APPROACH, VariationalAnsatzApproach):
#             plot_variational_error_scaling(results_data, MODEL, ERROR_TYPE, MU_VALUE)
#             # plot_variational_wavefunction(results_data, MODEL, MU_VALUE)