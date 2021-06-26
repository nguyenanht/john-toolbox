"""
API reference documentation for the example `john_toolbox` package.
"""
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)
import logging
from john_toolbox.utils.logger_config import setup_log_config

setup_log_config(is_dev=True)
LOGGER = logging.getLogger(__name__)
