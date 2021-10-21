import logging
from logging.handlers import TimedRotatingFileHandler
import pathlib
import sys

import logging.config
import os
import json


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
        path = "/work/john_toolbox/utils/logging_dev.json"
    else:
        path = "/work/john_toolbox/utils/logging.json"
    if os.path.exists(path):
        with open(path, "rt") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)

    return
