from .search_hugface import Huggingface
from .search_civitai import Civitai
from ..setup.base_config import Basic_config


class Config_Mix(
    Huggingface,
    Civitai,
    Basic_config
    ):
    #fix MMO error
    def __init__(self):
        super().__init__()
