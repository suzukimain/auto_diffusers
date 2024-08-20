# auto_diffusers

[![GitHub license](https://img.shields.io/badge/license-Apache-blue.svg)](
https://github.com/suzukimain/auto_diffusers/blob/main/LICENSE)



<details>
  <summary>Viewer</summary>
  <img src=https://visit-counter.vercel.app/counter.png?page=https://github.com/suzukimain/auto_diffusers/main&c=00ffff&ff=flat&tb=viewer:%20&s=20>  
</details>


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
from auto_diffusers import run_search(
model_path = run_search(<keyword>, auto=True, download=False)
pipe = StableDIffusionPipeline.from_single_file(model_path)
```



##  Description<a name = "Description"></a>

> seach_word (< keyword >)
  * type
    * string
  * Input available

    1.url
    
    2.directory or file

    3.Keywords to search
    
      * Example:
    
        1.anything"

        2.diffusion 1.5"
    
    4.huggingface path 

       * Format
          * < creator > / < repo >

> auto
  * desc
    * Minimize user input by automatically selecting the most highly rated models when searching for models.
  * type
    * bool
  * default
    * True

>  download
  * desc
    * Returns the path where the file was downloaded and saved.
  * type
    * bool
  * default
    * False


> model_type
  * desc
    * Valid only in Civitai.
  * type
    * string
  * default
    * "Checkpoint"
  * Input available
    1. Checkpoint
    2. TextualInversion
    3. Hypernetwork
    4. AestheticGradient
    5. LORA
    6. Controlnet
    7. Poses

> return_path
  * desc
    * Whether to return only the path or not. If false, returns [model_path , status_dict].
    
  * type
    * bool
  * default
    * False

> branch  
  * type
    * string
  * default
    * "main"

> local_file_only
  * type
    * bool
  * default
    * False

## License<a name = "License"></a>
In accordance with [BSD-3-Clause license](LICENSE)



## Acknowledgement<a name = "Acknowledgement"></a>

I have used open source resources and free tools in the creation of this project.

I would like to take this opportunity to thank the open source community and those who provided free tools.


