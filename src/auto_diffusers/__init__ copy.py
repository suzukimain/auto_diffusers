import os
import gc
import re
import sys
import glob
import time
import datetime
import torch
import warnings
import random
import imageio
import requests
import inspect
import difflib
import threading
import importlib
import diffusers
import urllib
import safetensors
import transformers
import numpy as np
import multiprocessing
from requests import HTTPError
from urllib import request
from torch import Generator
from base64 import b64encode
from IPython.display import display, Markdown
from PIL import Image,PngImagePlugin
from diffusers import StableDiffusionPipeline, AutoencoderKL, schedulers
from transformers import pipeline, CLIPTokenizer, AutoTokenizer, AutoModelForCausalLM
from IPython.display import display, HTML
from huggingface_hub import hf_hub_download
from multiprocessing import Process, Manager
from diffusers import logging as df_logging
from transformers import logging as tf_logging

#from model_path import pipeline_setup as pipe_set
#import pipe_set.model_set as path_set

#from utils.device_config import device_set

df_logging.set_verbosity_error()
tf_logging.set_verbosity_error()

warnings.filterwarnings("ignore")


#device = device_set.device_type_check()

def check_url(url) -> bool:
    "Determine if URL is valid"
    flag = True
    try:
        f = urllib.request.urlopen(url)
        f.close()
    except:
        flag = False
    return flag
