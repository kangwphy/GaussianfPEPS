import sub
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize
import numpy.polynomial.polynomial as P
import pandas as pd
import json
import os
import argparse
import time
# ==============================================================================
# PART 5: EXECUTION (Modified)
# ==============================================================================
if __name__ == '__main__':

    # --- 1. Setup Argument Parser ---
    # Model parameters
    parser = argparse.ArgumentParser(description="Run quantum state approximation calculations.")
    parser.add_argument('--model', type=str, default='2D', choices=['1D', '2D'], help='Model dimension (1D or 2D).')
    parser.add_argument('--error_type', type=str, default='energy', choices=['energy', 'fidelity'], help='Error type to optimize.')
    parser.add_argument('--t1', type=float, default=1.0, help='t1 parameter (NN hopping).')
    parser.add_argument('--t2', type=float, default=0.0, help='t2 parameter (NNN hopping).')
    parser.add_argument('--mu', type=float, default=0.0, help='Chemical potential (mu).')

    parser.add_argument(
        '--optimizer', 
        type=str, 
        default='Nelder-Mead',
        choices=['Nelder-Mead', 'L-BFGS-B', 'SLSQP', 'COBYLA', 'Powell'],
        help='Optimization method for the Variational Ansatz approach.'
    )
    parser.add_argument(
        '--approach', 
        type=str, 
        default='Ration',
        choices=['Ration', 'Variation'],
        help='Choosing approach.'
    )
    parser.add_argument(
        '--start_order', 
        type=int, 
        nargs='?', # 使参数可选
        default=1, 
        help='Start value for the order (inclusive).'
    )
    args = parser.parse_args()

    OPTIMIZER_NAME = args.optimizer
    START_ORDER = args.start_order
    ERROR_TYPE = args.error_type
    startt = time.time()
    
    # --- 2. Instantiate Model (Needed for Path) ---
    # Store model params from args. Note: mu is passed to run(), not init
    MODEL_PARAMS = {'t1': args.t1, 't2': args.t2, 'mu': args.mu}

    if args.model == '1D':
        MODEL = sub.Model1D(**MODEL_PARAMS)
        MODEL.INTEGRATION_LIMITS = (1e-3, np.pi) # Keep your override
    else:
        MODEL = sub.Model2D(**MODEL_PARAMS)
        MODEL.INTEGRATION_LIMITS = ((1e-3, np.pi), (1e-3, np.pi)) # Keep your override

    print(f"Running Model: {MODEL.NAME} with t1={args.t1}, t2={args.t2}, mu={args.mu}")

    # --- 3. Setup Directory Structure (NEW) ---
    
    # 1. Model Name (e.g., "1d-tight-binding")
    model_name_str = MODEL.NAME.lower().replace(' ', '-')
    
    # 2. Model Dependent String (e.g., "t1_1.0_t2_0.0_mu_0.0")
    model_dep_str = f"t1_{args.t1}_t2_{args.t2}_mu_{args.mu}"
    
    # 3. Optimizer Name (e.g., "Nelder-Mead")
    optimizer_str = OPTIMIZER_NAME
    
    # 4. Build base path
    if args.approach == "Variation":
        base_output_path = os.path.join(model_name_str, model_dep_str, optimizer_str)
    else:
        base_output_path = os.path.join(model_name_str, model_dep_str,"RationalApproxiation")
    # 5. Create final data/ and fig/ directories
    DATA_DIR = os.path.join(base_output_path, 'data')
    FIGS_DIR = os.path.join(base_output_path, 'fig')
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)
    
    print(f"Using optimizer: {OPTIMIZER_NAME}")
    print(f"Saving all outputs to: {base_output_path}")

    # Define global precision settings
    EPS_ABS = 1e-12
    EPS_REL = 1e-12

    # --- 4. Setup Integrator and Approach ---
    EPS_ABS = 1e-12
    EPS_REL = 1e-12
    integrator_manager = sub.IntegrationManager(epsabs=EPS_ABS, epsrel=EPS_REL)
    if args.approach == "Variation":
        APPROACH = sub.VariationalAnsatzApproach(integrator_manager)
    else:
        APPROACH = sub.RationalApproximationApproach(integrator_manager)
        
    
    # --- 5. Setup Orders ---
    if isinstance(APPROACH, sub.RationalApproximationApproach):
        ORDERS_TO_RUN = range(1, 20)
        results_data = APPROACH.run(
            model=MODEL, 
            error_type=ERROR_TYPE, 
            orders=ORDERS_TO_RUN, 
            mu_val=args.mu, # Pass mu from args
            # optimizer=OPTIMIZER_NAME, 
            # figs_dir=FIGS_DIR
            )
    elif isinstance(APPROACH, sub.VariationalAnsatzApproach):
        if MODEL.DIM == 1:
            # ORDERS_TO_RUN = [(4, 4), (6, 6), (8, 8)]
            ORDERS_TO_RUN = [(n, n) for n in range(START_ORDER,START_ORDER+1)]
        elif MODEL.DIM == 2:
            # ORDERS_TO_RUN = [(n, n) for n in range(1,10)]
            ORDERS_TO_RUN = [(START_ORDER, START_ORDER)]
        results_data = APPROACH.run(
            model=MODEL, 
            error_type=ERROR_TYPE, 
            orders=ORDERS_TO_RUN, 
            mu_val=args.mu, # Pass mu from args
            optimizer=OPTIMIZER_NAME, 
            figs_dir=FIGS_DIR
            )

   # --- 6. Run Calculation ---
    
    # --- 7. Save Results with Simplified Filenames ---
    # The path now contains all metadata, so the filename can be simpler.
    order_str = f"o{START_ORDER}"
    base_filename = f"{ERROR_TYPE}_{order_str}"

    # Create the header for the text file
    header = (f"{MODEL.NAME} {APPROACH.NAME} ({OPTIMIZER_NAME}) \n"
              f"Results for {ERROR_TYPE}, order={START_ORDER}\n"
              f"Params: t1={args.t1}, t2={args.t2}, mu={args.mu}")

    # Save to .txt and .csv in the new DATA_DIR
    sub.save_results_to_txt(os.path.join(DATA_DIR, f"{base_filename}.txt"), results_data, header)
    sub.save_results_to_dataframe(os.path.join(DATA_DIR, f"{base_filename}.csv"), results_data, header)
    
    # --- 8. Plotting ---
    if results_data:
        if isinstance(APPROACH, sub.RationalApproximationApproach):
            sub.plot_rational_approx_error_scaling(results_data, MODEL, ERROR_TYPE, args.mu, figs_dir=FIGS_DIR)
        elif isinstance(APPROACH, sub.VariationalAnsatzApproach):
            # Pass the base filename to plotting functions so they can save
            # with the same simple name (e.g., "energy_o5.png")
            sub.plot_variational_error_scaling(results_data, MODEL, ERROR_TYPE, args.mu, figs_dir=FIGS_DIR, base_filename=base_filename)

    # --- 9. Finish ---
    elapsed_time = time.time() - startt
    
    # 格式化输出结果
    print("\n==============================================")
    print(f"Run complete for: {base_output_path}")
    print(f"Total Elapsed Time: {elapsed_time:.4f} s")
    print("==============================================")