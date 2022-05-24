import logging
from logging.handlers import TimedRotatingFileHandler
import pathlib
import sys

import logging.config
import os
import json


ACCEPTED_LEVEL_MODE = {
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "CRITICAL": logging.CRITICAL,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def setup_log_config(is_dev=False, level=None):
    """Setup logging configuration.

    Parameters
    ----------
    is_dev: boolean
        if True, the dev configuration log file is used (more readable for developers)
        you can activate the dev mode passing 'True' to ENGINE_LOGS_DEV into
        the 'engine-service.yml' from brain_docker
    level: str, default=None
        parameters to override

    """
    if is_dev:
        path = "/work/john_toolbox/utils/logging_dev.json"
    else:
        path = "/work/john_toolbox/utils/logging.json"
    if os.path.exists(path):
        with open(path, "rt") as f:
            config = json.load(f)

        if level is not None:
            if level.upper() in ACCEPTED_LEVEL_MODE.keys():

                config["root"]["level"] = level.upper()
                config["handlers"]["console"]["level"] = level.upper()
        print(f"level logging = {config['root']['level']}")
        logging.config.dictConfig(config)
    else:

        if level is not None:
            custom_level = ACCEPTED_LEVEL_MODE.get(level.upper())
            if custom_level is not None:
                logging.basicConfig(level=custom_level)
                print(f"level logging = {level.upper()}")
            else:
                logging.basicConfig(level=logging.INFO)
                print("level logging = INFO")

        else:
            logging.basicConfig(level=logging.INFO)
            print("level logging = INFO")
    return
