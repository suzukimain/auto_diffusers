# auto_diffusers

[![GitHub license](https://img.shields.io/badge/license-Apache-blue.svg)](
https://github.com/suzukimain/auto_diffusers/blob/main/LICENSE)


<div>
  <img src=https://visit-counter.vercel.app/counter.png?page=https://github.com/suzukimain/auto_diffusers/main&c=00ffff&ff=flat&tb=viewer:%20&s=20>  
</div>


>CONTENTS
+ [About The Project](#About_The_Project)
+ [How to use](#How_to_use)
+ [Description](#Description)
+ [License](#License)
+ [Acknowledgement](#Acknowledgement)

## About The Project<a name = "About_The_Project"></a>
Enhance the functionality of diffusers.
* Search models from huggingface and Civitai. 
(etc..)


##  How to use<a name = "How_to_use"></a>

```python
pip install diffusers
pip install git+https://github.com/suzukimain/auto_diffusers

from diffusers import StableDiffusionPipeline
from auto_diffusers import run_search

model_path = run_search(<keyword>, auto=True, download=False)
pipe = StableDIffusionPipeline.from_single_file(model_path)
```

##  Description<a name = "Description"></a>
> Arguments of `run_ search`
> 
| Name           | Type   | Default     | Input Available  | Description |
|:--------------:|:------:|:-----------:|:----------------:|:--------------------------------------------------------:|
| search_word    | string | ー          | [Details](#search-word) | Keywords to search models |
| auto           | bool   | True        | ー                | Minimize user input by selecting the highest-rated models. |
| download       | bool   | False       | ー                | Returns the path where the file was downloaded. |
| model_type     | string | "Checkpoint"| 1. `Checkpoint`<br>2. `TextualInversion`<br>3. `Hypernetwork`<br>4. `AestheticGradient`<br>5. `LORA`<br>6. `Controlnet`<br>7. `Poses` | Valid only in Civitai. |
| return_path    | bool   | True        | ー                | Returns only the path or `[model_path, status_dict]`. |
| branch         | string | "main"      | ー                | Specify the branches of huggingface and civitai. |
| local_file_only| bool   | False       | ー                | Search local folders only. |


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


## License<a name = "License"></a>
In accordance with [BSD-3-Clause license](LICENSE)



## Acknowledgement<a name = "Acknowledgement"></a>

I have used open source resources and free tools in the creation of this project.

I would like to take this opportunity to thank the open source community and those who provided free tools.


