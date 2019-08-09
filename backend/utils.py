"""Utilities for logging."""
import os
import logging
import time

if os.getenv('DEBUG'):
    level = logging.DEBUG
else:
    level = logging.ERROR
logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        )
logger = logging.getLogger(__name__)


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger = logging.getLogger(method.__name__)
        logger.debug('{} {:.3f} sec'.format(method.__name__, te-ts))
        return result

    return timed
