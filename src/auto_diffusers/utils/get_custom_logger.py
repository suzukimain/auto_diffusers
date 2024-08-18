import os
import logging


def custom_logger():
    """
    When debugging, put “DEBUG” in the environment variable.
    """
    debug = os.environ.get("DEBUG",False)
    format = '%(levelname)s:<cell line: %(lineno)d> <funcName: %(funcName)s>: %(message)s'
    logger = logging.getLogger(__name__)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter(format)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler) 
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)
    return logger

logger = custom_logger()
