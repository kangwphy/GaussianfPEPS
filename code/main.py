"""
Main execution script for Gaussian fPEPS calculations.
This script provides a command-line interface for running quantum state approximation
calculations using either Rational Approximation or Variational Ansatz approaches.

Usage:
    # Run from the parent directory:
    python gitsyn/main.py --model 1D --approach Ration --start_order 5

    # Or run from anywhere with Python path setup:
    cd /path/to/2DGaussian_fPEPS
    python -m gitsyn.main --model 1D --approach Ration
"""

import os
import sys
import argparse
import time
import numpy as np

# Add the parent directory to path to enable imports when running from gitsyn directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import all required functionality from the reorganized sub module
from gitsyn.sub import *

def setup_directories(model_name, task, params, approach, optimizer_name, start_order):
    """
    Create directory structure for saving results and figures.

    Args:
        model_name: Name of the model
        task: Task/domain type
        params: Model parameters dictionary
        approach: Approach type (Rational or Variation)
        optimizer_name: Name of optimizer
        start_order: Starting order

    Returns:
        tuple: (DATA_DIR, FIGS_DIR) paths
    """
    # Model name (e.g., "1d-tight-binding")
    model_name_str = model_name.lower().replace(' ', '-')

    # Model parameters string (e.g., "t1_1.0_t2_0.0_t3_0.0_mu_0.0")
    model_dep_str = f"t1_{params['t1']}_t2_{params['t2']}_t3_{params['t3']}_mu_{params['mu']}"

    # Build base path
    if approach == "Variation":
        base_output_path = os.path.join("data", task, model_name_str, model_dep_str, optimizer_name)
    else:
        base_output_path = os.path.join("data", task, model_name_str, model_dep_str, "RationalApproximation")

    # Create final data/ and fig/ directories
    data_dir = os.path.join(base_output_path, 'data')
    figs_dir = os.path.join(base_output_path, 'fig')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    return data_dir, figs_dir

def setup_integration_regions_2d(model, task):
    """
    Setup integration regions for 2D models based on task type.

    Args:
        model: Model instance
        task: Task/domain type
    """
    model.DOMAIN_TYPE = task

    if task == 'Rectangle':
        low_b = 1e-3
        model.INTEGRATION_REGIONS = [
            ((low_b, np.pi), lambda x: low_b, lambda x: np.pi)
        ]

    elif task == 'L-Shape':
        low_b = 1e-3
        model.INTEGRATION_REGIONS = [
            ((low_b, np.pi/4), lambda x: np.pi/4, lambda x: np.pi),  # R1
            ((np.pi/4, np.pi), lambda x: low_b, lambda x: np.pi)    # R2
        ]

    elif task == 'SquareMinusCircle':
        low_b = 0
        r_min = np.pi/4
        theta_min = low_b
        theta_mid = np.pi/4
        theta_max = np.pi/2 - low_b

        model.INTEGRATION_REGIONS = [
            # R1: theta from [low_b, pi/4]
            ((theta_min, theta_mid),
             lambda theta: r_min,
             lambda theta: np.pi / np.cos(theta)),
            # R2: theta from [pi/4, pi/2 - low_b]
            ((theta_mid, theta_max),
             lambda theta: r_min,
             lambda theta: np.pi / np.sin(theta))
        ]

    elif task == 'Sector':
        model.INTEGRATION_REGIONS = [
            ((0, np.pi/2), lambda theta: 0, lambda theta: np.pi)
        ]

    elif task == 'AnnularSector':
        r_min = np.pi / 2.0
        r_max = np.pi * 3.0 / 2.0
        theta_min = 0
        theta_max = np.pi/2

        print(f"Defining AnnularSector: r in [{r_min}, {r_max}], theta in [{theta_min}, {theta_max}]")

        model.INTEGRATION_REGIONS = [
            ((theta_min, theta_max),
             lambda theta: r_min,
             lambda theta: r_max)
        ]

    elif task == 'DiagonalStrip':
        r1 = np.pi / 2.0
        r2 = np.pi * 3.0/2.0

        def g_func(kx):
            return np.maximum(0, r1 - kx)

        def h_func(kx):
            return np.minimum(np.pi, r2 - kx)

        x_min = np.maximum(0, r1 - np.pi)
        x_max = np.minimum(np.pi, r2 - 0)

        print(f"Defining DiagonalStrip: {r1} <= kx+ky <= {r2}")
        print(f"Integration range kx: [{x_min:.4f}, {x_max:.4f}]")

        model.INTEGRATION_REGIONS = [
            ((x_min, x_max), g_func, h_func)
        ]

    elif task == 'DiagonalStrip2':
        r1 = np.pi * 3.0/2.0
        r2 = np.pi / 2.0

        def g_func(kx):
            return np.minimum(np.pi, r1 + kx)

        def h_func(kx):
            return np.maximum(0, r2 + kx)

        x_min = 0.0
        x_max = -np.pi

        print(f"Defining DiagonalStrip2 (R2): {r1} <= ky-kx <= {r2}")
        print(f"Integration range kx: [{x_min:.4f}, {x_max:.4f}]")

        model.INTEGRATION_REGIONS = [
            ((x_min, x_max), g_func, h_func)
        ]

    else:
        raise ValueError(f"Domain '{task}' not recognized for 2D coordinates.")


def main():
    """Main execution function."""

    # --- 1. Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Run quantum state approximation calculations.")

    # Model parameters
    parser.add_argument('--model', type=str, default='2D', choices=['1D', '2D'],
                       help='Model dimension (1D or 2D).')
    parser.add_argument('--error_type', type=str, default='energy',
                       choices=['energy', 'fidelity'], help='Error type to optimize.')
    parser.add_argument('--t1', type=float, default=1.0, help='t1 parameter (NN hopping).')
    parser.add_argument('--t2', type=float, default=0.0, help='t2 parameter (NNN hopping).')
    parser.add_argument('--t3', type=float, default=0.0, help='t3 parameter (NNN hopping).')
    parser.add_argument('--mu', type=float, default=0.0, help='Chemical potential (mu).')

    # Approach and optimization parameters
    parser.add_argument('--task', type=str, default='test',
                       help='Domain type for integration.')
    parser.add_argument('--optimizer', type=str, default='Nelder-Mead',
                       choices=['Nelder-Mead', 'L-BFGS-B', 'SLSQP', 'COBYLA', 'Powell'],
                       help='Optimization method for the Variational Ansatz approach.')
    parser.add_argument('--approach', type=str, default='Ration',
                       choices=['Ration', 'Variation'], help='Calculation approach.')
    parser.add_argument('--start_order', type=int, default=1,
                       help='Start value for the order (inclusive).')
    parser.add_argument('--max_orders', type=int, default=3,
                       help='Maximum number of orders to calculate.')

    args = parser.parse_args()

    # Extract parameters
    optimizer_name = args.optimizer
    start_order = args.start_order
    error_type = args.error_type
    start_time = time.time()

    # --- 2. Instantiate Model ---
    if args.model == '1D':
        # 1D model doesn't have t3 parameter
        model_params = {'t1': args.t1, 't2': args.t2, 'mu': args.mu}
        model = Model1D(**model_params)

        # Setup 1D integration limits
        if args.task == "half":
            model.INTEGRATION_LIMITS = (0.25*np.pi, 0.75*np.pi)
            print("MODEL", model.get_bz_volume(), model.INTEGRATION_LIMITS)
        else:
            model.INTEGRATION_LIMITS = (1e-3, np.pi)

    else:  # 2D model
        # 2D model has t3 parameter
        model_params = {'t1': args.t1, 't2': args.t2, 't3': args.t3, 'mu': args.mu}
        model = Model2D(**model_params)
        setup_integration_regions_2d(model, args.task)

    print(f"Running Model: {model.NAME} with t1={args.t1}, t2={args.t2}, t3={args.t3}, mu={args.mu}, domain={args.task}")

    # --- 3. Setup Directory Structure ---
    # Use the same parameters for directory naming regardless of model type
    dir_params = {'t1': args.t1, 't2': args.t2, 't3': args.t3, 'mu': args.mu}
    data_dir, figs_dir = setup_directories(
        model.NAME, args.task, dir_params, args.approach, optimizer_name, start_order
    )

    print(f"Using optimizer: {optimizer_name}")
    print(f"Saving all outputs to: {os.path.dirname(data_dir)}")

    # --- 4. Setup Integrator and Approach ---
    eps_abs = 1e-12
    eps_rel = 1e-12
    integrator_manager = IntegrationManager(epsabs=eps_abs, epsrel=eps_rel)

    if args.approach == "Variation":
        approach = VariationalAnsatzApproach(integrator_manager)
    else:
        approach = RationalApproximationApproach(integrator_manager)

    # --- 5. Setup Orders and Run Calculation ---
    if isinstance(approach, RationalApproximationApproach):
        orders_to_run = range(start_order, start_order + args.max_orders)
        results_data = approach.run(
            model=model,
            error_type=error_type,
            orders=orders_to_run,
            mu_val=args.mu
        )
    else:  # Variational approach
        if model.DIM == 1:
            orders_to_run = [(n, n) for n in range(start_order, start_order + args.max_orders)]
        else:  # 2D
            orders_to_run = [(start_order, start_order)]

        results_data = approach.run(
            model=model,
            error_type=error_type,
            orders=orders_to_run,
            mu_val=args.mu,
            optimizer=optimizer_name,
            figs_dir=figs_dir
        )

    # --- 6. Save Results ---
    order_str = f"o{start_order}"
    if args.approach == "Variation":
        base_filename = f"{error_type}_{order_str}"
    else:
        base_filename = f"{error_type}"

    header = (f"{model.NAME} {approach.NAME} ({optimizer_name})\n"
              f"Results for {error_type}, order={start_order}\n"
              f"Params: t1={args.t1}, t2={args.t2}, t3={args.t3}, mu={args.mu}")

    # Save to .txt and .csv files
    save_results_to_txt(os.path.join(data_dir, f"{base_filename}.txt"), results_data, header)
    save_results_to_dataframe(os.path.join(data_dir, f"{base_filename}.csv"), results_data, header)

    # --- 7. Plotting ---
    if results_data:
        if isinstance(approach, RationalApproximationApproach):
            plot_rational_approx_error_scaling(results_data, model, error_type, args.mu, figs_dir=figs_dir)
        elif isinstance(approach, VariationalAnsatzApproach):
            plot_variational_error_scaling(results_data, model, error_type, args.mu, figs_dir=figs_dir)

    # --- 8. Finish ---
    elapsed_time = time.time() - start_time

    print("\n" + "="*50)
    print(f"Run complete for: {os.path.dirname(data_dir)}")
    print(f"Total Elapsed Time: {elapsed_time:.4f} s")
    print("="*50)


if __name__ == '__main__':
    main()