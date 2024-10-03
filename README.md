# auto_diffusers


<p>
    <a href="https://github.com/suzukimain/auto_diffusers/blob/main/LICENSE"><img alt="GitHub release" src="https://img.shields.io/badge/license-BSD%203--Clause-blue.svg?style=social"></a>
</p>
<p>
    <a href="https://pepy.tech/project/auto_diffusers"><img alt="GitHub release" src="https://static.pepy.tech/badge/auto_diffusers"></a>
    <a href="https://github.com/suzukimain/auto_diffusers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/suzukimain/auto_diffusers.svg"></a>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=suzukimain.auto_diffusers" alt="Visitor Badge">
</p>


>CONTENTS
+ [About The Project](#About_The_Project)
+ [How to use](#How_to_use)
+ [Example](#Example)
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
pip install auto_diffusers

from diffusers import StableDiffusionPipeline
from auto_diffusers import model_search


path = model_search(
           <keyword>,
           auto = True,
           model_format="diffusers",
           download = False
           )
pipe = StableDiffusionPipeline.from_pretrained(path)

# or

path = model_search(
           <keyword>,
           auto = True,
           model_format="single_file",
           download = False
           )
pipe = StableDIffusionPipeline.from_single_file(path)
```

##  Example<a name = "Example"></a>

```python
pip install --quiet diffusers
pip install --quiet auto_diffusers

from diffusers import StableDiffusionPipeline
from IPython.display import display
from auto_diffusers import model_search

model_path = model_search(
                 "Any",
                 auto=True,
                 model_format="diffusers",
                 download=False
                 )
pipe = StableDiffusionPipeline.from_pretrained(model_path).to("cuda")

image = pipe("Mt. Fuji").images[0]

print(f"model_path: {model_path}")
display(image)
```

##  Description<a name = "Description"></a>
> Arguments of `model_search`
> 
| Name           | Type   | Default     | Input Available  | Description |
|:--------------:|:------:|:-----------:|:----------------:|:--------------------------------------------------------:|
| search_word    | string | ー          | [Details](#search-word) | Keywords to search models |
| auto           | bool   | True        | ー                | Minimize user input by selecting the highest-rated models. |
| download       | bool   | False       | ー                | Returns the path where the file was downloaded. |
| model_format   | string | "single_file" | `all`,<br> `diffusers`,<br> `single_file`| Specifies the format of the model. [Details](#model_format) |
| model_type     | string | "Checkpoint"| `Checkpoint`,<br>`TextualInversion`,<br>`Hypernetwork`,<br>`AestheticGradient`,<br>`LORA`,<br>`Controlnet`,<br>`Poses` | Valid only in Civitai. |
| return_path    | bool   | True        | ー                | Returns only the path or `[model_path, status_dict]`. |
| branch         | string | "main"      | ー                | Specify the branches of huggingface and civitai. |
| local_file_only| bool   | False       | ー                | Search local folders only.<br>**In the case of `auto`, files with names similar to `search_word` will be given priority.** |



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


<a id="model_format"></a>
<details open>
<summary>model_format</summary>

| Argument                     | Description                                                            |
| :--------------------------: | :--------------------------------------------------------------------: |
| all                          | In auto, `multifolder diffusers format checkpoint` takes precedence    |                                      
| single_file                  | Only `single file checkpoint` are searched.  |
| diffusers                    | Search only for `multifolder diffusers format checkpoint`<br>**Note that only the huggingface is searched for, since it is not in civitai.**    |

</details>


## License<a name = "License"></a>
In accordance with [BSD-3-Clause license](LICENSE)



## Acknowledgement<a name = "Acknowledgement"></a>

I have used open source resources and free tools in the creation of this project.

I would like to take this opportunity to thank the open source community and those who provided free tools.


