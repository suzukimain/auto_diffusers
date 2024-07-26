def custom_logger(DEBUG=False):
    format = '%(levelname)s:<cell line: %(lineno)d> <funcName: %(funcName)s>: %(message)s'
    logger = logging.getLogger(__name__)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter(format)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if DEBUG:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)
    return logger
