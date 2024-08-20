from .setup.Base_config import Basic_config
from .model_path.Perform_path_search import search_path
from .utils.run_download import download
from .checker.get_checkpoint_type import checkpoint_type
import search_path.run_search as search_for_path

device = Basic_config().device_type_check()