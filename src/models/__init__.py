import logging

from ultralytics.utils import LOGGER as UL_LOGGER

# disable the logs of ultralytics
UL_LOGGER.setLevel(logging.ERROR)
