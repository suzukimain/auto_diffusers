import re
import jax
import importlib
from device_config import device_set
from build.lib.utils.method_until import Get_method
from get_custom_logger import logger


class basic_config(device_set,Get_method):
    @staticmethod
    def module_version(module_name):
        try:
            version = importlib.metadata.version(module_name)
            return re.match(r"^\d+\.\d+\.\d+", version).group(0)
        except importlib.metadata.PackageNotFoundError:
            return None
    