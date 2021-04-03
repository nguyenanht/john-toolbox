import logging
from logging.handlers import TimedRotatingFileHandler
import pathlib
import sys

FORMATTER = logging.Formatter(
    "%(levelname)s - %(asctime)s|%(name)s - %(funcName)s:%(lineno)d — %(message)s",
    "%H:%M:%S",
)

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent

loggers = {}


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_logger(*, logger_name):
    """Get logger with prepared handlers."""
    global loggers

    if loggers.get(logger_name):
        return loggers.get(logger_name)
    else:
        logger = logging.getLogger(logger_name)

        # logger.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.addHandler(get_console_handler())
        logger.propagate = False
        loggers[logger_name] = logger

    return loggers[logger_name]
