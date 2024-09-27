#main

from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass

import diffusers
from diffusers import (
    StableDiffusionPipeline,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    TextToVideoZeroPipeline,
    FlaxStableDiffusionPipeline,
    FlaxStableDiffusionImg2ImgPipeline,
    FlaxStableDiffusionInpaintPipeline,
    )

from model_path.perform_path_search import Search_cls
from model_path.mix_class import Basic_config


@dataclass
class AutoPipe_data:
    pipe_dict = {
        "torch":{
            "base" : StableDiffusionPipeline,
            "txt2img" : AutoPipelineForText2Image,
            "img2img" : AutoPipelineForImage2Image,
            "inpaint" : AutoPipelineForInpainting,
            "txt2video" : TextToVideoZeroPipeline,
        },
        "flax" : {
            "base" : FlaxStableDiffusionPipeline,
            "txt2img" : FlaxStableDiffusionPipeline,
            "img2img" : FlaxStableDiffusionImg2ImgPipeline,
            "inpaint" : FlaxStableDiffusionInpaintPipeline,
            "txt2video" : None,
        }

    }



class AutoPipeline(Search_cls,Basic_config):
    def __init__(
            self,
            search_word: str,
            pipe_type = "txt2img",
            auto: Optional[bool] = True,
            priority: Optional[str] = "hugface",
            branch: Optional[str] = "main",
            local_file_only: Optional[bool] = False,
            ):
        super().__init__()
        self.device = self.device_type_check()
    

    def pipeline_type(
            self,
            cls_or_name
            ):
        if isinstance(str, cls_or_name):
            if hasattr(diffusers, cls_or_name):
                return getattr(diffusers, cls_or_name)
            else:
                candidate = self.max_temper(cls_or_name,dir(diffusers))
                error_txt = f"Maybe {candidate}?" if candidate else ""
                raise ValueError(f"{cls_or_name} is not in diffusers.{error_txt}")
        
        elif hasattr(diffusers, cls_or_name.__name__):
            return cls_or_name
        
        else:
            candidate = self.max_temper(cls_or_name,dir(diffusers))
            error_txt = f"Maybe {candidate}?" if candidate else ""
            raise ValueError(f"{cls_or_name} is not in diffusers.{error_txt}")





