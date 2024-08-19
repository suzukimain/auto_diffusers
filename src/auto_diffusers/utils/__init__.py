import os
import re
import jax
import importlib
from auto_diffusers.setup.device_config import device_set
from auto_diffusers.utils.get_custom_logger import logger
from .method_until import basic_config