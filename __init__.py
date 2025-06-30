# modules/__init__.py
"""
ACC Sales Intelligence System - Modules Package
"""

__version__ = "1.0.0"
__author__ = "ACC Sales Intelligence Team"

# Import all modules to make them available
try:
    from . import data_loader
    from . import sales_dashboard
    from . import customer_segmentation
    from . import sales_forecasting
    from . import salesperson_performance
    from . import discount_analysis
    from . import bu_benchmarking
    from . import product_insights
except ImportError as e:
    # Handle missing modules gracefully
    print(f"Warning: Some modules could not be imported: {e}")
