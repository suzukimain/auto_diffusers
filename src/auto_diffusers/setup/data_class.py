
from dataclasses import dataclass

@dataclass
class data_config:
    Config_file="model_index.json"

    VALID_URL_PREFIXES = ["https://huggingface.co/", "huggingface.co/", "hf.co/", "https://hf.co/"]
    exts =  [".safetensors", ".ckpt",".bin"]

    model_dict = {
            "stable diffusion-v2.1" : "stabilityai/stable-diffusion-2-1",
            "waifu diffusion-v1.4": "hakurei/waifu-diffusion",
            "Anything-v3.0": "Linaqruf/anything-v3.0",
            "anything-midjourney-v-4-1": "Joeythemonster/anything-midjourney-v-4-1",
            "Anything-v4.5": "shibal1/anything-v4.5-clone",
            "AB4.5_AC0.2": "aioe/AB4.5_AC0.2",
            "basil_mix": "nuigurumi/basil_mix",
            "Waifu-Diffusers": "Nilaier/Waifu-Diffusers",
            "Double-Exposure-Diffusion": "joachimsallstrom/Double-Exposure-Diffusion",
            "openjourney-v4": "prompthero/openjourney-v4",
            "ACertainThing": "JosephusCheung/ACertainThing",
            "Counterfeit-V2.0": "gsdf/Counterfeit-V2.0",
            "Counterfeit-V2.5": "gsdf/Counterfeit-V2.5",
            "chilled_remix":"chilled_remix",
            "chilled_reversemix":"chilled_reversemix",
            "7th_Layer": "syaimu/7th_test",
            "EimisAnimeDiffusion_1.0v": "eimiss/EimisAnimeDiffusion_1.0v",
            "JWST-Deep-Space-diffusion" : "dallinmackay/JWST-Deep-Space-diffusion",
            "Riga_Collection": "natsusakiyomi/Riga_Collection",
            "sd-db-epic-space-machine" : "rabidgremlin/sd-db-epic-space-machine",
            "spacemidj" : "Falah/spacemidj",
            "anime-kawai-diffusion": "Ojimi/anime-kawai-diffusion",
            "Realistic_Vision_V2.0": "SG161222/Realistic_Vision_V2.0",
            "nasa-space-v2" : "sd-dreambooth-library/nasa-space-v2-768",
            "meinamix_meinaV10": "namvuong96/civit_meinamix_meinaV10",
            "loliDiffusion": "JosefJilek/loliDiffusion",
            }
    exclude =  ["safety_checker/model.safetensors",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
                "text_encoder/model.safetensors",
                "unet/diffusion_pytorch_model.fp16.safetensors",
                "text_encoder/model.fp16.safetensors",
                "vae/diffusion_pytorch_model.fp16.safetensors",
                "safety_checker/model.fp16.safetensors",

                "safety_checker/model.ckpt",
                "unet/diffusion_pytorch_model.ckpt",
                "vae/diffusion_pytorch_model.ckpt",
                "text_encoder/model.ckpt",
                "text_encoder/model.fp16.ckpt",
                "safety_checker/model.fp16.ckpt",
                "unet/diffusion_pytorch_model.fp16.ckpt",
                "vae/diffusion_pytorch_model.fp16.ckpt"]

    Auto_pipe_class=[
            "AutoPipelineForText2Image",
            "AutoPipelineForImage2Image",
            "AutoPipelineForInpainting",
     ]

    Error_M1 = (
        '''
        Could not load URL.
        Format:"https://huggingface.co/<repo_name>/<model_name>/blob/main/<path_to_file>"
        EX1: "https://huggingface.co/gsdf/Counterfeit-V3.0/blob/main/Counterfeit-V3.0.safetensors"
        EX2: "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt"
        '''
        )

    Error_M2= (
        '''
        Could not load hugface_path.
        Format: <repo_name>/<model_name>"
        EX1: "Linaqruf/anything-v3.0"
        EX2: "stabilityai/stable-diffusion-2-1"

        Suport_model:

                "stable diffusion-v2.1"
                "waifu diffusion-v1.4"
                "Anything-v3.0"
                "anything-midjourney-v-4-1"
                "Anything-v4.5"
                "AB4.5_AC0.2"
                "basil_mix"
                "Waifu-Diffusers"
                "Double-Exposure-Diffusion"
                "openjourney-v4"
                "ACertainThing"
                "Counterfeit-V2.0"
                "Counterfeit-V2.5"
                "7th_Layer"
                "EimisAnimeDiffusion_1.0v"
                "Riga_Collection"
                "anime-kawai-diffusion"
                "Realistic_Vision_V2.0"
                "meinamix_meinaV10"
                "loliDiffusion"
                ''')

    Error_M3 = ('''
                The specified path could not be recognized. Please try the following
                ・Check that the path to the file exists.
                ・Check that there is no whitespace in the path.
                ・Check if there are any special symbols such as "\" or "." and other special symbols (may not be recognized).
                ''')
    config_file_list = [
        "preprocessor_config.json",
        "config.json",
        "model.fp16.safetensors",
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.fp16.bin",
        "scheduler_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
        "diffusion_pytorch_model.bin",
        "diffusion_pytorch_model.fp16.bin",
        "diffusion_pytorch_model.fp16.safetensors",
        "diffusion_pytorch_model.non_ema.bin",
        "diffusion_pytorch_model.non_ema.safetensors",
        "diffusion_pytorch_model.safetensors",
        ]
    
    def __init__(self):
        self.single_file_only = False
        self.exclude_untested_model = True
        self.return_dict = {
            "search_word" : "",
            "url_or_path" : "",
            "load_type" : "",# "" or "from_single_file" or "from_pretrained"
            "single_file" : False,
            "local" : False,
            "repo_id":"",
            "viersion_id":"",
            "filename":"",
            "file_id": "",
            "file_size": 0,
            "fp": "",                
            "download_url": "",
            }
        