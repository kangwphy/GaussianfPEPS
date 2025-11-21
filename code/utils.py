import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as P
import pandas as pd
import json
import os
from scipy.integrate import quad, dblquad


# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')


class IntegrationManager:
    """A wrapper to manage global precision settings for integration."""

    def __init__(self, epsabs=1e-16, epsrel=1e-16):
        """
        Initialize the integration manager.

        Args:
            epsabs (float): Absolute tolerance for integration
            epsrel (float): Relative tolerance for integration
        """
        self.epsabs = epsabs
        self.epsrel = epsrel

    def quad(self, func, a, b, args=(), **kwargs):
        """
        Wrapper for scipy.integrate.quad with pre-set precision.

        Args:
            func: Function to integrate
            a, b: Integration limits
            args: Additional arguments for the function
            **kwargs: Additional keyword arguments

        Returns:
            tuple: (result, error_estimate)
        """
        return quad(func, a, b, args=args, epsabs=self.epsabs, epsrel=self.epsrel, **kwargs)

    def dblquad(self, func, a, b, gfun, hfun, args=(), **kwargs):
        """
        Wrapper for scipy.integrate.dblquad with pre-set precision.

        Args:
            func: Function to integrate
            a, b: Outer integration limits
            gfun, hfun: Inner integration limit functions
            args: Additional arguments for the function
            **kwargs: Additional keyword arguments

        Returns:
            tuple: (result, error_estimate)
        """
        return dblquad(func, a, b, gfun, hfun, args=args, epsabs=self.epsabs, epsrel=self.epsrel, **kwargs)


def save_results_to_txt(filename, data, header):
    """
    Saves human-readable results to a text file.
    Correctly handles data from both Rational and Variational approaches.

    Args:
        filename (str): Output file path
        data: Results data from approach
        header (str): Header text for the file
    """
    with open(filename, 'w') as f:
        f.write(f"{header}\n=========================\n")

        # Check if data is from RationalApproach (a dict with 'errors')
        if isinstance(data, dict) and "errors" in data:
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
    Correctly handles data from both Rational and Variational approaches.

    Args:
        filename (str): Output CSV file path
        data: Results data from approach
        header (str): Header text for documentation
    """
    if not data:
        print("No data to save.")
        return

    print(f"Processing results for: {header}")

    # Convert list/dict to a DataFrame
    df = pd.DataFrame(data)

    # Pre-process the DataFrame based on the approach
    if 'error_init' in df.columns:
        # Variational data (list of dicts)
        if 'orders' in df.columns:
            df[['order_u', 'order_v']] = pd.DataFrame(df['orders'].tolist(), index=df.index)
            df = df.drop(columns=['orders'])

        # Convert coefficient lists to JSON strings
        for col in ['u_coeffs', 'v_coeffs', 'u_coeffs_init', 'v_coeffs_init']:
            if col in df.columns:
                df[col] = df[col].apply(json.dumps)

        # Reorder columns for clarity
        desired_order = [
            'order_u', 'order_v', 'error_init', 'error', 'success',
            'u_coeffs_init', 'v_coeffs_init', 'u_coeffs', 'v_coeffs'
        ]
        final_columns = [col for col in desired_order if col in df.columns]
        df = df[final_columns]

    elif 'errors' in df.columns:
        # Rational data (dict of lists)
        df = df[['orders', 'errors']]

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Results successfully saved to '{filename}' in CSV format.")


def plot_rational_approx_error_scaling(data, model, error_type, mu_val, figs_dir='figs'):
    """
    Plots error scaling for the Rational Approximation approach.

    Args:
        data: Results data with 'orders' and 'errors'
        model: Physical model instance
        error_type (str): Type of error ('energy' or 'fidelity')
        mu_val (float): Chemical potential value
        figs_dir (str): Directory for saving figures
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    title = f'{model.NAME} {error_type.capitalize()} Error Scaling (Rational Approx., μ={mu_val})'
    fig.suptitle(title, fontsize=16)

    orders_np = np.array(data['orders'])
    errors_np = np.array(data['errors'])
    label = f"{error_type.capitalize()} Error"

    axes[0].plot(orders_np, errors_np, 'o-', label=label)
    axes[0].set_yscale('log')
    axes[0].set_title('log(Error) vs. n')
    axes[0].set_xlabel('Order (n)')
    axes[0].set_ylabel('Error')

    axes[1].plot(np.sqrt(orders_np), errors_np, 's-', label=label)
    axes[1].set_yscale('log')
    axes[1].set_title('log(Error) vs. sqrt(n)')
    axes[1].set_xlabel('sqrt(n)')

    axes[2].plot(orders_np, errors_np, '^-', label=label)
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_title('log(Error) vs. log(n)')
    axes[2].set_xlabel('Order (n)')

    for ax in axes:
        ax.grid(True, which="both", ls="--")
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = os.path.join(figs_dir, f"{model.NAME.lower()}_rational_{error_type}_scaling_mu{mu_val}.png")
    plt.savefig(filename)
    plt.show()


def plot_variational_error_scaling(results, model, error_type, mu_val, figs_dir='figs'):
    """
    Plots error scaling for the Variational Ansatz approach.

    Args:
        results: Results data from variational approach
        model: Physical model instance
        error_type (str): Type of error ('energy' or 'fidelity')
        mu_val (float): Chemical potential value
        figs_dir (str): Directory for saving figures
    """
    if not results:
        return

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
    plt.savefig(filename)
    plt.show()


def plot_wavefunction_comparison(variational_result, rational_coeffs, model, mu_val, error_type, figs_dir='figs'):
    """
    Generates a 3x2 plot comparing the final optimized variational wavefunction
    against the rational approximation (initial guess).

    This function supports both 1D models and 2D models. For 2D models,
    it plots the data along the diagonal path from (0,0) to (pi,pi).

    Args:
        variational_result: Results from variational optimization
        rational_coeffs: Coefficients from rational approximation
        model: Physical model instance
        mu_val (float): Chemical potential value
        error_type (str): Type of error ('energy' or 'fidelity')
        figs_dir (str): Directory for saving figures
    """
    # Extract Data and Coefficients
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

    # Setup Momentum Path based on Model Dimension
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
        x_axis_label = r'Momentum Path $|k|$ from $(0,0)$ to $(\pi,\pi)$'
    else:
        print(f"Wavefunction comparison plot is not implemented for DIM={model.DIM} models.")
        return

    # Helper Function to Calculate Plotting Data
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

    # Calculate Data for Both Initial and Final States
    uk_init, vk_init, prob_u_init, prob_v_init, integrand_init = get_plotting_data(initial_coeffs)
    uk_final, vk_final, prob_u_final, prob_v_final, integrand_final = get_plotting_data(var_coeffs)

    # Create the Figure and Subplots
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
    axes[2, 0].set_xlabel(x_axis_label)
    axes[2, 0].set_ylabel('Amplitude')

    # Plot 6: v(k) component
    axes[2, 1].plot(x_axis_data, vk_init.real, 'r--', label='Initial Guess (Rational)', alpha=0.9)
    axes[2, 1].plot(x_axis_data, vk_final.real, 'g-', label='Variational (Optimized)', lw=2)
    axes[2, 1].set_title('Wavefunction Component $v(k)$')
    axes[2, 1].set_xlabel(x_axis_label)
    axes[2, 1].set_ylabel('Amplitude')

    # Final Touches and Saving
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

    savename = os.path.join(figs_dir, f"{model.NAME.lower()}_variational_{error_type}_comparison_u{order_u}v{order_v}_mu{mu_val}.png")
    plt.savefig(savename)
    print(f"Comparison plot saved to: {savename}")
    plt.close(fig)