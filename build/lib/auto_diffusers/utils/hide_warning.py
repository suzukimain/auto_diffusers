import warnings
from diffusers import logging as df_logging
from transformers import logging as tf_logging

df_logging.set_verbosity_error()
tf_logging.set_verbosity_error()

warnings.filterwarnings("ignore")