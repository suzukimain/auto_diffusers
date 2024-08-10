import os
import torch
import requests
import yaml
from io import BytesIO
from utils.get_custom_logger import logger
from utils import basic_config


class checkpoint_type(basic_config):
    def __init__(self):
        pass

    
    def checkpoint_type_get(
            self,
            checkpoint_path_or_dict,
            config_files = None,
            original_config_file = None,
            model_type = None,
            is_upscale = False
            ):
        """
        NOTE:
        Return SD if you do not know, because an error may occur if None is used.

        About model_type:
        The model_type itself will be left for possible use at some point.
        """

        from_safetensors = False
        checkpoint = None
        if isinstance(checkpoint_path_or_dict, str):
            if os.path.isfile(checkpoint_path_or_dict):
                from_safetensors: bool = self.check_for_safetensors(checkpoint_path_or_dict)
                if from_safetensors:
                    from safetensors.torch import load_file as safe_load
                    checkpoint = safe_load(checkpoint_path_or_dict, device=self.device)
                else:
                    checkpoint = torch.load(checkpoint_path_or_dict, map_location=self.device)
            elif os.path.isdir(checkpoint_path_or_dict):
                model_index_path = os.path.join(checkpoint_path_or_dict,"model_index.json")
                if os.path.isfile(model_index_path):
                    with open(model_index_path,"r") as loaded_model_index:
                        cls_name = loaded_model_index["_class_name"]
                        #Fixed in due course.
                        if "XL" in cls_name:
                            return "SDXL"
                        else:
                            return "SD"

        elif isinstance(checkpoint_path_or_dict, dict):
            checkpoint = checkpoint_path_or_dict
        else:
            raise TypeError(f"checkpoint_path_or_dict: {checkpoint_path_or_dict}")

        if "global_step" in checkpoint:
            global_step = checkpoint["global_step"]
        else:
            global_step = None

        while "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if original_config_file is None:
            key_name_v2_1 = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
            key_name_sd_xl_base = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias"
            key_name_sd_xl_refiner = "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias"
            #is_upscale = pipeline_class == StableDiffusionUpscalePipeline
            config_url = None
            # model_type = "v1"
            if config_files is not None and "v1" in config_files:
                original_config_file = config_files["v1"]
            else:
                config_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"

            if key_name_v2_1 in checkpoint and checkpoint[key_name_v2_1].shape[-1] == 1024:
                # model_type = "v2"
                if config_files is not None and "v2" in config_files:
                    original_config_file = config_files["v2"]
                else:
                    config_url = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"
                if global_step == 110000:
                    # v2.1 needs to upcast attention
                    upcast_attention = True

            elif key_name_sd_xl_base in checkpoint:
                # only base xl has two text embedders
                if config_files is not None and "xl" in config_files:
                    original_config_file = config_files["xl"]
                else:
                    config_url = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"

            elif key_name_sd_xl_refiner in checkpoint:
                # only refiner xl has embedder and one text embedders
                if config_files is not None and "xl_refiner" in config_files:
                    original_config_file = config_files["xl_refiner"]
                else:
                    config_url = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_refiner.yaml"

            if is_upscale:
                config_url = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/x4-upscaling.yaml"

            if config_url is not None:
                try:
                    original_config_file = BytesIO(requests.get(config_url).content)
                except:
                    logger.error(f"Could not download the Config_file to find out the model type from the following URL: {config_url}")
                    if model_type is None:
                        logger.warning("model_type is set to None")
                    return model_type
            else:
                with open(original_config_file, "r") as f:
                    original_config_file = f.read()
        else:
            with open(original_config_file, "r") as f:
                original_config_file = f.read()

        original_config = yaml.safe_load(original_config_file)

        if (
            model_type is None
            and "cond_stage_config" in original_config["model"]["params"]
            and original_config["model"]["params"]["cond_stage_config"] is not None
        ):

            model_type = original_config["model"]["params"]["cond_stage_config"]["target"].split(".")[-1]
            return "SD"

        elif model_type is None and original_config["model"]["params"]["network_config"] is not None:
            if original_config["model"]["params"]["network_config"]["params"]["context_dim"] == 2048:
                model_type = "SDXL"
            else:
                model_type = "SDXL-Refiner"
            return "SDXL"

        else:
            if model_type is None:
                logger.warning("model_type is set to None")
            return model_type
