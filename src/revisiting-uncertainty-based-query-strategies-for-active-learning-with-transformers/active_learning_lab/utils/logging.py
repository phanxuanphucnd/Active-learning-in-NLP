import logging
import sys


def setup_logger():

    logger = logging.getLogger()
    logger.handlers = []

    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s',
                                  datefmt='%H:%M:%S')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel('INFO')

    return logger
