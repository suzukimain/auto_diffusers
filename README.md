# auto_diffusers


<p>
    <a href="https://github.com/suzukimain/auto_diffusers/blob/main/LICENSE"><img alt="GitHub release" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=social"></a>
</p>
<p>
    <a href="https://pepy.tech/project/auto_diffusers"><img alt="GitHub release" src="https://static.pepy.tech/badge/auto_diffusers"></a>
    <a href="https://github.com/suzukimain/auto_diffusers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/suzukimain/auto_diffusers.svg"></a>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=suzukimain.auto_diffusers" alt="Visitor Badge">
</p>


>CONTENTS
+ [About The Project](#About_The_Project)
+ [How to use](#How_to_use)
+ [Description](#Description)
+ [License](#License)
+ [Acknowledgement](#Acknowledgement)

## About The Project<a name = "About_The_Project"></a>
Enhance the functionality of diffusers.
* Search models from huggingface and Civitai.


##  How to use<a name = "How_to_use"></a>

```python
pip install --quiet auto_diffusers
```

```python
from auto_diffusers import EasyPipelineForText2Image

# Search for Huggingface
pipe = EasyPipelineForText2Image.from_huggingface("any").to("cuda")
img = pipe("cat").images[0]
img.save("cat.png")


# Search for Civitai
pipe = EasyPipelineForText2Image.from_civitai("any").to("cuda")
image = pipe("cat").images[0]
image.save("cat.png")

```

---

```python
from auto_diffusers import (
    search_huggingface,
    search_civitai,
) 

# Search Lora
Lora = search_civitai(
    "Keyword_to_search_Lora",
    model_type="LORA",
    base_model = "SD 1.5",
    download=True,
    )
# Load Lora into the pipeline.
pipeline.load_lora_weights(Lora)


# Search TextualInversion
TextualInversion = search_civitai(
    "EasyNegative",
    model_type="TextualInversion",
    base_model = "SD 1.5",
    download=True
)
# Load TextualInversion into the pipeline.
pipeline.load_textual_inversion(TextualInversion, token="EasyNegative")

```

## Description<a name = "Description"></a>
> Arguments of `EasyPipeline.from_huggingface`

| Name                  | Type                            | Default        | Input Available   | Description                                                                                                          |
|:---------------------:|:------------------------------:|:--------------:|:-----------------:|:--------------------------------------------------------------------------------------------------------------------:|
| pretrained_model_or_path | str or os.PathLike            | ー             | ー                | Keywords to search models                                                                                            |
| checkpoint_format     | string                          | "single_file"  | `single_file`,<br>`diffusers`,<br>`all` | The format of the model checkpoint.                                                             |
| pipeline_tag          | string                          | None           | ー                 | Tag to filter models by pipeline.                                                                                    |
| torch_dtype           | str or torch.dtype              | None           | ー                 | Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the dtype is automatically derived from the model's weights. |
| force_download        | bool                            | False          | ー                 | Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist. |
| cache_dir             | Union[str, os.PathLike]         | None           | ー                 | Path to a directory where a downloaded pretrained model configuration is cached if the standard cache is not used.   |
| token                 | str or bool                     | None           | ー                 | The token to use as HTTP bearer authorization for remote files.                                                      |

<a id="model_format"></a>
<details open>
<summary>model_format</summary>

| Argument                     | Description                                                            |
| :--------------------------: | :--------------------------------------------------------------------: |                               
| single_file                  | Only `single file checkpoint` are searched.  |
| diffusers                    | Search only for `multifolder diffusers format checkpoint    |

</details>


<a id="Other_Arguments"></a>
<details close>
<summary>Other_Arguments</summary>
  
| Name                  | Type                            | Default        | Input Available   | Description                                                                                                          |
|:---------------------:|:------------------------------:|:--------------:|:-----------------:|:--------------------------------------------------------------------------------------------------------------------:|
| proxies               | Dict[str, str]                  | None           | ー                 | A dictionary of proxy servers to use by protocol or endpoint.                                                        |
| output_loading_info   | bool                            | False          | ー                 | Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.              |
| local_files_only      | bool                            | False          | ー                 | Whether to only load local model weights and configuration files or not.                                             |
| revision              | str                             | "main"         | ー                 | The specific model version to use.                                                                                   |
| custom_revision       | str                             | "main"         | ー                 | The specific model version to use when loading a custom pipeline from the Hub or GitHub.                             |
| mirror                | str                             | None           | ー                 | Mirror source to resolve accessibility issues if you’re downloading a model in China.                                |
| device_map            | str or Dict[str, Union[int, str, torch.device]] | None | ー            | A map that specifies where each submodule should go.                                                           |
| max_memory            | Dict                            | None           | ー                 | A dictionary device identifier for the maximum memory.                                                               |
| offload_folder        | str or os.PathLike              | None           | ー                 | The path to offload weights if device_map contains the value `"disk"`.                                               |
| offload_state_dict    | bool                            | True           | ー                 | If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM.                |
| low_cpu_mem_usage     | bool                            | Depends on torch version | ー        | Speed up model loading only loading the pretrained weights and not initializing the weights.                         |
| use_safetensors       | bool                            | None           | ー                 | If set to `None`, the safetensors weights are downloaded if they're available and if the safetensors library is installed. |
| gated                 | bool                            | False          | ー                 | A boolean to filter models on the Hub that are gated or not.                                                         |
| kwargs                | dict                            | None           | ー                 | Can be used to overwrite load and saveable variables.                                                                |
| variant               | str                             | None           | ー                 | Load weights from a specified variant filename such as `"fp16"` or `"ema"`.                                          |

</details>



---

> [!TIP]
> **If an error occurs, insert the `token` and run again.**

> Arguments of `EasyPipeline.from_civitai`

| Name            | Type   | Default       | Input Available   | Description                                                                         |
|:---------------:|:------:|:-------------:|:-----------------:|:-----------------------------------------------------------------------------------:|
| search_word     | string | ー            | ー                 | Keywords to search models                                                             |
| model_type      | string | `Checkpoint`  | ー                 | The type of model to search for.                                                      |
| base_model      | string | None          | ー                 | Trained model tag (example:  `SD 1.5`, `SD 3.5`, `SDXL 1.0`)                          |
| download        | bool   | False         | ー                 | Whether to download the model.                                                        |
| force_download  | bool   | False         | ー                 | Whether to force the download if the model already exists.                            |
| cache_dir       | string, Path | None    | ー                 | Path to the folder where cached files are stored.                                     |
| resume          | bool   | False         | ー                 | Whether to resume an incomplete download.                                             |
| token           | string | None          | ー                 | API token for Civitai authentication.                                                 |
| skip_error      | bool   | False         | ー                 | Whether to skip errors and return None.                                               |




<a id="search-word"></a>
<details open>
<summary>search_word</summary>

| Type                         | Description                                                            |
| :--------------------------: | :--------------------------------------------------------------------: |
| keyword                      | Keywords to search model<br>                                           |
| url                          | Can be any URL other than huggingface or Civitai.                      |
| Local directory or file path | Search for files with the extensions: `.safetensors`, `.ckpt`, `.bin`  |
| huggingface path             | The following format: `< creator > / < repo >`                         |

</details>


<a id="model_type"></a>
<details open>
<summary>model_type</summary>
    
| Input Available | 
| :--------------------------------------: |
|  `Checkpoint`,<br>`TextualInversion`,<br>`Hypernetwork`,<br>`AestheticGradient`,<br>`LORA`,<br>`Controlnet`,<br>`Poses` |

</details>


## License<a name = "License"></a>
In accordance with [Apache-2.0 license](https://github.com/suzukimain/auto_diffusers/blob/main/LICENSE)


## Acknowledgement<a name = "Acknowledgement"></a>

I have used open source resources and free tools in the creation of this project.

I would like to take this opportunity to thank the open source community and those who provided free tools.
