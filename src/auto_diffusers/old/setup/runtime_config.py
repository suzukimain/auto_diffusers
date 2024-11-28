import re
import requests
import logging
from importlib import metadata

from .device_config import device_set
from ..utils.get_custom_logger import custom_logger


class Runtime_func(device_set):
    def __init__(self):
        """
        NOTE:
        Functions that should be included in the `Basic_config` class are needed in the `install_packages` class, so these functions are separated into the `runtime_func` class.
        This is because the `install_packages` class is executed before the `Basic_config` class is executed, so these functions are separated and inherited later.
        """
        self.device_type = self.device_type_check()
        self.logger = custom_logger()
        super().__init__()


    def module_version(self,module_name):
        try:
            version = metadata.version(module_name)
            return re.match(r"^\d+\.\d+\.\d+", version).group(0) #type: ignore
        except metadata.PackageNotFoundError:
            return None


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