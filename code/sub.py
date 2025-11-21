"""
Main module for Gaussian fPEPS calculations.
This module provides backward compatibility by importing all functions and classes
from the organized package structure.
"""

# Import all functions and classes from the new modular structure
from .models import BaseModel, Model1D, Model2D
from .rational import xi, p, g, RationalApproximationApproach
from .variational import VariationalAnsatzApproach
from .utils import IntegrationManager, save_results_to_txt, save_results_to_dataframe, \
                  plot_rational_approx_error_scaling, plot_variational_error_scaling, \
                  plot_wavefunction_comparison

# Import plotting utilities for backward compatibility
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

# Re-export everything for backward compatibility
__all__ = [
    # Core rational approximation functions
    'xi', 'p', 'g',

    # Models
    'BaseModel', 'Model1D', 'Model2D',

    # Approaches
    'RationalApproximationApproach', 'VariationalAnsatzApproach',

    # Utilities
    'IntegrationManager', 'save_results_to_txt', 'save_results_to_dataframe',
    'plot_rational_approx_error_scaling', 'plot_variational_error_scaling',
    'plot_wavefunction_comparison'
]

# For legacy code that might import from sub directly
# All the code is now organized in separate modules, but can still be accessed through here

# Example usage section (commented out, but kept for reference)
"""
# ==============================================================================
# PART 5: EXECUTION EXAMPLE
# ==============================================================================
if __name__ == '__main__':
    # Define global precision settings
    EPS_ABS = 1e-16
    EPS_REL = 1e-16

    # Create the integration manager instance
    integrator_manager = IntegrationManager(epsabs=EPS_ABS, epsrel=EPS_REL)

    # Choose approach
    APPROACH = RationalApproximationApproach(integrator_manager)
    # or
    APPROACH = VariationalAnsatzApproach(integrator_manager)

    # Choose model
    MODEL = Model1D()
    # MODEL = Model2D()

    # Choose parameters
    ERROR_TYPE = 'fidelity'
    # ERROR_TYPE = 'energy'
    MU_VALUE = 0.0

    # Set orders based on approach
    if isinstance(APPROACH, RationalApproximationApproach):
        ORDERS_TO_RUN = range(1, 100)
    elif isinstance(APPROACH, VariationalAnsatzApproach):
        if MODEL.DIM == 1:
            ORDERS_TO_RUN = [(n, n) for n in range(15, 16)]
        elif MODEL.DIM == 2:
            ORDERS_TO_RUN = [(2, 2), (3, 3)]

    # Run calculations
    results_data = APPROACH.run(
        model=MODEL, error_type=ERROR_TYPE, orders=ORDERS_TO_RUN, mu_val=MU_VALUE,
    )

    # Save results
    header = f"{MODEL.NAME} {APPROACH.NAME} Results for {ERROR_TYPE}, mu={MU_VALUE}"
    filename = f"{MODEL.NAME.lower()}_{APPROACH.NAME.replace(' ','')}_{ERROR_TYPE}_mu{MU_VALUE}.txt"
    save_results_to_txt("data/" + filename, results_data, header)

    # Plot results if available
    if results_data:
        if isinstance(APPROACH, RationalApproximationApproach):
            plot_rational_approx_error_scaling(results_data, MODEL, ERROR_TYPE, MU_VALUE)
        elif isinstance(APPROACH, VariationalAnsatzApproach):
            plot_variational_error_scaling(results_data, MODEL, ERROR_TYPE, MU_VALUE)
"""