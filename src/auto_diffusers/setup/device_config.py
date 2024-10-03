import torch
import jax

from ..utils.get_custom_logger import custom_logger


class device_set:
    def __init__(self):
        self.device_type = self.device_type_check()
        self.logger = custom_logger()


    def device_type_check(self):
        if torch.cuda.is_available():
            return "cuda"
        else:
            device_type = jax.devices()[0].device_kind
            if "TPU" in device_type:
                return "TPU"
            else:
                return "cpu"
            
    
    def count_device(self):
        return jax.device_count()


    def is_TPU(self):
        if self.device_type_check() == "TPU":
            return True
        else:
            return False
