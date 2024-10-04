import os


from .mix_class import Config_Mix


class Search_cls(Config_Mix):
    def __init__(self):
        super().__init__()


    def __call__(
            self,
            seach_word,
            auto=True,
            download=False,
            model_type="Checkpoint",
            model_format = "single_file",
            branch = "main",
            priority = "hugface",
            local_file_only = False,
            return_path = True,
            exclude_untested_model = False
            ):
        self.seach_word = seach_word
        self.auto = auto
        self.download = download
        self.model_type = model_type
        self.branch = branch
        self.local_file_only=local_file_only
        self.model_format =model_format
        self.return_path = return_path
        self.exclude_untested_model = exclude_untested_model

        result = self.model_set(
                  model_select = seach_word,
                  auto = auto,
                  download = download,
                  model_format = model_format,
                  model_type = model_type,
                  branch = branch,
                  priority = priority,
                  local_file_only = local_file_only,
                  return_path = return_path
                  )
        return result
        

    def File_search(
            self,
            search_word,
            auto = True
            ):
        """
        only single file
        """
        search_path=""
        paths = []
        for root, dirs, files in os.walk("/"):
            for file in files:
                if any(file.endswith(ext) for ext in self.exts):
                    path = os.path.join(root, file)
                    if path not in self.exclude:
                        if not path.startswith("/root/.cache"):
                            paths.append(path)
        num_path=len(paths)
        if not num_path:
            raise FileNotFoundError("\033[33mModel File not found\033[0m")
        elif not auto:
            print(f"{num_path} candidate model files found.")
            for s, path in enumerate(paths, 1):
                print(f"{s}: {path}")
            num = int(input(f"Please enter a number(1〜{num_path}): "))
            if 1 <= num <= len(paths):
                search_path=(paths[num-1])
                print(f"Selected model file: {search_path}\n")
            else:
                raise TypeError(f"\033[33mOnly natural numbers in the following range are valid : (1〜{len(paths)})\033[0m")
        else:
            search_path = self.max_temper(
                search_word = search_word,
                search_list = paths
                )
        return search_path


    def model_set(
            self,
            model_select,
            auto = True,
            download = False,
            model_format = "single_file",
            model_type = "Checkpoint",
            branch = "main",
            priority = "hugface",
            local_file_only = False,
            return_path = True
            ):
        """
        parameter:
        model_format:
            one of the following: "all","diffusers","single_file"
        return:
        if path_only is false
        [model_path:str, {base_model_path: str,single_file: bool}]
        """

        if not model_type  in ["Checkpoint", "TextualInversion", "LORA", "Hypernetwork", "AestheticGradient", "Controlnet", "Poses"]:
            raise TypeError(f'Wrong argument. Valid values are "Checkpoint", "TextualInversion", "LORA", "Hypernetwork", "AestheticGradient", "Controlnet", "Poses". What was passed on {model_type}')
        
        if not model_format in ["all","diffusers","single_file"]:
            raise TypeError('The model_format is valid only for one of the following: "all","diffusers","single_file"')

        self.return_dict["local"] = True if download or local_file_only else False
        
        model_path = model_select
        file_path = ""
        if model_select in self.model_dict:
            model_path_to_check = self.model_dict[model_select]
            _check_url = f"https://huggingface.co/{model_path_to_check}"
            if self.is_url_valid(_check_url):
                model_select = model_path_to_check
                self.return_dict["url_or_path"] = _check_url


        if local_file_only:
            model_path = self.File_search(
                search_word = model_select,
                auto = auto
                )
            self.return_dict["single_file"] = True
            self.return_dict["url_or_path"] = model_path
            self.return_dict["load_type"] = "from_single_file"

        elif model_select.startswith("https://huggingface.co/"):
            if not self.is_url_valid(model_select):
                raise ValueError(self.Error_M1)
            else:
                if download:
                    model_path = self.run_hf_download(model_select)
                else:
                    model_path = model_select

                self.return_dict["single_file"] = True
                self.return_dict["url_or_path"] = model_path
                repo,file_name = self.repo_name_or_path(model_select)
                if file_name:
                    self.return_dict["single_file"] = True
                    self.return_dict["load_type"] = "from_single_file"
                else:
                    self.return_dict["single_file"] = False
                    self.return_dict["load_type"] = "from_pretrained"


        elif model_select.startswith("https://civitai.com/"):
            model_path = self.civitai_model_set(
                    model_select=model_select,
                    auto=auto,
                    model_type=model_type,
                    download=download
                    )

        elif os.path.isfile(model_select):
            model_path = model_select
            self.return_dict["url_or_path"] = model_select
            self.return_dict["single_file"] = True
            self.return_dict["load_type"] = "from_single_file"
            self.return_dict["local"] = True


        elif os.path.isdir(model_select):
            if os.path.exists(os.path.join(model_select,self.Config_file)):
                model_path = model_select
                self.return_dict["url_or_path"] = model_select
                self.return_dict["single_file"] = False
                self.return_dict["load_type"] = "from_pretrained"
                self.return_dict["local"] = True
            else:
                raise FileNotFoundError(f"model_index.json not found in {model_select}")

        elif model_select.count("/") == 1:
            if auto and self.diffusers_model_check(model_select):
                if download:
                    model_path = self.run_hf_download(model_select)
                    self.return_dict["single_file"] = False
                else:
                    model_path = model_select
                    self.return_dict["single_file"] = False
                self.return_dict["load_type"] = "from_pretrained"

            elif auto and (not self.hf_model_check(model_select)):
                raise ValueError(f'The specified repository could not be found, please try turning off "auto" (model_select:{model_select})')
            else:
                file_path=self.file_name_set(model_select,auto,model_type)
                if file_path == "_hf_no_model":
                    raise ValueError("Model not found")
                elif file_path == "_DFmodel":
                    if download:
                        model_path= self.run_hf_download(model_select)
                    else:
                        model_path = model_select

                    self.return_dict["url_or_path"] = model_path
                    self.return_dict["single_file"] = False
                    self.return_dict["load_type"] = "from_pretrained"
                    
                else:
                    hf_model_path=f"https://huggingface.co/{model_select}/blob/{branch}/{file_path}"
                    
                    if download:
                        model_path = self.run_hf_download(hf_model_path)
                    else:
                        model_path = hf_model_path
                    self.return_dict["single_file"] = True
                    self.return_dict["url_or_path"] = model_path
                    self.return_dict["load_type"] = "from_single_file"

        else:
            if priority == "hugface":
                model_path = self.hf_model_set(
                    model_select=model_select,
                    auto=auto,
                    model_format=model_format,
                    model_type=model_type,
                    download=download
                    )
                if model_path == "_hf_no_model":
                    model_path = self.civitai_model_set(
                        model_select=model_select,
                        auto=auto,
                        model_type=model_type,
                        download=download
                        )
                    if model_path == "_civitai_no_model":
                        raise ValueError("No models matching the criteria were found.")
                
            else:
                model_path = self.civitai_model_set(
                    model_select=model_select,
                    auto=auto,
                    model_type=model_type,
                    download=download
                    )
                if model_path == "_civitai_no_model":
                    model_path = self.hf_model_set(
                        model_select = model_select,
                        auto = auto,
                        model_format=model_format,
                        model_type=model_type,
                        download=download
                        )
                    if model_path == "_hf_no_model":
                        raise ValueError("No models matching the criteria were found.")
                
        #It is not called, but should not be deleted because it is updating the dictionary.
        update_model_path = self.check_func_hist(key="model_path",value=model_path)
        if return_path:
            return model_path
        else:
            return [model_path, self.return_dict]
        

    def hf_model_set(
            self,
            model_select,
            auto,
            model_format,
            model_type,
            download
            ):
        model_path = ""
        model_name = self.model_name_search(
            model_name=model_select,
            auto_set=auto,
            model_format=model_format)
        #hf->civit
        if not model_name == "_hf_no_model":
            file_path = self.file_name_set(
                model_select=model_name,
                auto=auto,
                model_format=model_format,
                model_type=model_type)
            if file_path == "_DFmodel":
                if download:
                    model_path = self.run_hf_download(model_name,branch=self.branch)
                else:
                    model_path = model_name
                self.return_dict["single_file"] = False
                self.return_dict["load_type"] = "from_pretrained"

            else:
                hf_model_path = f"https://huggingface.co/{model_name}/blob/{self.branch}/{file_path}"
                if download:
                    model_path = self.run_hf_download(hf_model_path)
                else:
                    model_path = hf_model_path
                self.return_dict["single_file"] = True
                self.return_dict["load_type"] = "from_single_file"

            return model_path
        else:
            return "_hf_no_model"


    def civitai_model_set(
            self,
            model_select,
            auto,
            model_type,
            download
            ):

        model_url, model_path = self.civitai_download(
            model_select,
            auto,
            model_type,
            download=download)
        
        
        self.return_dict["single_file"] = True
        self.return_dict["url_or_path"] = model_url
        
        if download:
            self.return_dict["load_type"] = "from_single_file"
        else:
            self.return_dict["load_type"] = ""
        return model_path
