__version__ = "2.0.25"

from .pipeline_easy import (
    search_huggingface,
    search_civitai,
    load_pipeline_from_single_file,
    EasyPipelineForText2Image,
    EasyPipelineForImage2Image,
    EasyPipelineForInpainting,
)