import jax

from ..utils.get_custom_logger import custom_logger


class device_set:
    def __init__(self):
        self.device_type = self.device_type_check()
        self.logger = custom_logger()


    def device_type_check(self):
        _device_type = jax.devices()[0].device_kind
        if "TPU" in _device_type:
            return "TPU"
        elif "cpu" in _device_type:
            return "cpu"
        else:
            return "cuda"


    def  extra_device_set(self):
        device_type = self.device_type_check()
        
        if device_type == "TPU":
            #import torch_xla.core.xla_model as xm
            #device = xm.xla_device()
            device = device_type
        else:
            device = device_type
        return device

    
    def count_device(self):
        return jax.device_count()


    def is_TPU(self):
        if self.device_type_check() == "TPU":
            return True
        else:
            return False
