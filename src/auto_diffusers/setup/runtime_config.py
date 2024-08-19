import jax
import re
import requests
import importlib
import logging
from ..utils.get_custom_logger import custom_logger




class Runtime_func:
    def __init__(self):
        """
        NOTE:
        Functions that should be included in the `basic_config` class are needed in the `install_packages` class, so these functions are separated into the `runtime_func` class.
        This is because the `install_packages` class is executed before the `basic_config` class is executed, so these functions are separated and inherited later.
        """
        self.device_type = self.device_type_check()
        self.logger = custom_logger()



    def module_version(self,module_name):
        try:
            version = importlib.metadata.version(module_name)
            return re.match(r"^\d+\.\d+\.\d+", version).group(0)
        except importlib.metadata.PackageNotFoundError:
            return None


    def device_type_check(self):
        _device_type = jax.devices()[0].device_kind
        if "TPU" in _device_type:
            return "TPU"
        elif "cpu" in _device_type:
            return "cpu"
        else:
            return "cuda"


    def is_url_valid(self,url) -> bool:
        response = requests.head(url)
        try:
            response.raise_for_status()
        except requests.RequestException:
            return False
        else:
            return True
        finally:
            self.logger.debug(f"response.status_code: {response.status_code}")


    def DEBUG(self,debug:bool):
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)