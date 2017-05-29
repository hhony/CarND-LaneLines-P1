import logging

DEBUG = False
ch = logging.StreamHandler()


def set_debug_flag():
    global DEBUG, logger
    DEBUG = True
    logger = get_logger(__name__)


def get_logger(name=__name__):
    '''
    Retrieves a named logger.
    DEBUG is set to False by default.
    :param name: default name
    :return: logging instance
    '''
    global DEBUG, logger, ch
    logger = logging.getLogger(name)

    if DEBUG:
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(levelname)s] %(funcName)s: - %(message)s")
        ch.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(levelname)s] %(message)s")
        ch.setFormatter(formatter)
        logger.setLevel(logging.INFO)

    logger.addHandler(ch)
    return logger


logger = get_logger(__name__)
