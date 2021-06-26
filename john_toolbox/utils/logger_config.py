import logging
from logging.handlers import TimedRotatingFileHandler
import pathlib
import sys

import logging.config
import os
import json

#
# FORMATTER = logging.Formatter(
#     "%(levelname)s - %(asctime)s|%(name)s - %(funcName)s:%(lineno)d — %(message)s",
#     "%H:%M:%S",
# )
#
# PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
#
# loggers = {}
#
#
# def get_console_handler():
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setFormatter(FORMATTER)
#     return console_handler
#
#
# def get_logger(*, logger_name):
#     """Get logger with prepared handlers."""
#     global loggers
#
#     if loggers.get(logger_name):
#         return loggers.get(logger_name)
#     else:
#         logger = logging.getLogger(logger_name)
#
#         # logger.setLevel(logging.INFO)
#         logger.setLevel(logging.INFO)
#         if logger.hasHandlers():
#             logger.handlers.clear()
#
#         logger.addHandler(get_console_handler())
#         logger.propagate = False
#         loggers[logger_name] = logger
#
#     return loggers[logger_name]


def setup_log_config(is_dev=True):
    """Setup logging configuration.

    Parameters
    ----------
    is_dev: boolean
        if True, the dev configuration log file is used (more readable for developers)
        you can activate the dev mode passing 'True' to ENGINE_LOGS_DEV into
        the 'engine-service.yml' from brain_docker

    """
    if is_dev:
        path = "../john_toolbox/utils/logging_dev.json"
    else:
        path = ".../john_toolbox/utils/logging.json"
    if os.path.exists(path):
        with open(path, "rt") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)

    return
