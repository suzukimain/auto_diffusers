from .setup.base_config import Basic_config
from .utils.run_download import download
from .model_path.perform_path_search import Search_cls

# `run_search` is obsolete. Please use `model_search` instead.
model_search = Search_cls()
