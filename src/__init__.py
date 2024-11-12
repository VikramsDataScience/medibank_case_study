from pathlib import Path
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

# Relative imports
from .config import Config

# Model version
__version__ = 0.01

# Load storage paths from the config.py file
config = Config()
data_path = Path(config.data_path)
case_study_path = Path(config.case_study_path)
charts_reports_path = Path(config.charts_reports_path)

def suppress_statsmodels_warnings(func):
    """
    Boilerplate function that can be used as a Decorator to suppress 
    specific statsmodels ValueWarnings about frequency inference. 
    Can be used with any function that might trigger these warnings.
    """
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 
                                 message='No frequency information was provided',
                                 category=ValueWarning,
                                 module='statsmodels')
            return func(*args, **kwargs)
    return wrapper