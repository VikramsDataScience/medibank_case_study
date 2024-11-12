from pathlib import Path

# Relative imports
from .config import Config

# Model version
__version__ = 0.01

# Load storage paths from the config.py file
config = Config()
data_path = Path(config.data_path)
case_study_path = Path(config.case_study_path)
charts_reports_path = Path(config.charts_reports_path)
