"""
Gaussian fPEPS package for rational approximation and variational ansatz calculations.
"""

from .models import BaseModel, Model1D, Model2D
from .rational import xi, p, g, RationalApproximationApproach
from .variational import VariationalAnsatzApproach
from .utils import IntegrationManager, save_results_to_txt, save_results_to_dataframe, \
                  plot_rational_approx_error_scaling, plot_variational_error_scaling, \
                  plot_wavefunction_comparison

__all__ = [
    # Models
    'BaseModel', 'Model1D', 'Model2D',

    # Rational approximation
    'xi', 'p', 'g', 'RationalApproximationApproach',

    # Variational methods
    'VariationalAnsatzApproach',

    # Utilities
    'IntegrationManager',
    'save_results_to_txt',
    'save_results_to_dataframe',
    'plot_rational_approx_error_scaling',
    'plot_variational_error_scaling',
    'plot_wavefunction_comparison'
]