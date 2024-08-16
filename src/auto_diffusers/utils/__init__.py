import os
import re
import jax
import importlib
from device_config import device_set
from get_custom_logger import logger
from .method_until import basic_config