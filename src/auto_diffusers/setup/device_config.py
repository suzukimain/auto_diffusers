import jax
import tensorflow as tf

from ..utils.get_custom_logger import custom_logger


class device_set:
    def __init__(self):
        self.device_type = self.device_type_check()
        self.logger = custom_logger()
    
    def device_type_check(self):
        if tf.config.experimental.list_physical_devices('GPU'):
            device_type = "cuda"
        else:
            try:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPUの検出
                tf.tpu.experimental.initialize_tpu_system(tpu)
                device_type = "TPU"
            except ValueError:
                device_type = "cpu"
        return device_type


    def extra_device_type_check(self):
        _device_type = jax.devices()[0].device_kind
        if "TPU" in _device_type:
            return "TPU"
        elif "cpu" in _device_type:
            return "cpu"
        else:
            return "cuda"


    def extra_device_set(self):
        device_type = self.device_type_check()
        
        if device_type == "TPU":
            #import torch_xla.core.xla_model as xm
            #device = xm.xla_device()
            device = tf.distribute.cluster_resolver.TPUClusterResolver() # TPUの検出
            tf.tpu.experimental.initialize_tpu_system(device)
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
