import os
import re
from numpy import full
import requests
from requests import HTTPError

from diffusers import DiffusionPipeline #type: ignore
from huggingface_hub import (
    hf_hub_download, 
    HfApi
    )
from dataclasses import asdict
    

from ..setup.base_config import Basic_config


class Huggingface(Basic_config):
    def __init__(self):
        super().__init__()
        self.num_prints=20
        self.model_id=""
        self.model_name=""
        self.vae_name=""
        self.model_file=""
        self.diffuser_model=False
        self.check_choice_key = ""
        self.choice_number = -1
        self.file_path_dict={}
        self.special_file=""
        self.hf_repo_id = ""
        self.force_download = False
        self.hf_token = None
        self.hf_api = HfApi()


    def repo_name_or_path(self,model_name_or_path):
        pattern = r"([^/]+)/([^/]+)/(?:blob/main/)?(.+)"
        weights_name = None
        repo_id = None
        for prefix in self.VALID_URL_PREFIXES:
            model_name_or_path = model_name_or_path.replace(prefix, "")
        match = re.match(pattern, model_name_or_path)
        if not match:
            return repo_id, weights_name
        repo_id = f"{match.group(1)}/{match.group(2)}"
        weights_name = match.group(3)
        return repo_id, weights_name
    

    def _hf_repo_download(self,path,branch="main"):
        return DiffusionPipeline.download(
            pretrained_model_name = path,
            revision=branch,
            force_download=self.force_download,
            token=self.hf_token
            )


    def run_hf_download(
            self,
            url_or_path,
            branch="main",
            ) -> str:
        """
        retrun:
        os.path(str)
        """     
        model_file_path = ""
        if any(url_or_path.startswith(checked) for checked in self.VALID_URL_PREFIXES):
            if not self.is_url_valid(url_or_path):
                raise HTTPError(f"Invalid URL: {url_or_path}")
            hf_path, file_name = self.repo_name_or_path(url_or_path)
            self.logger.debug(f"url_or_path:{url_or_path}")
            self.logger.debug(f"hf_path: {hf_path} \nfile_name: {file_name}")
            if hf_path and file_name:
                #single_file = True
                model_file_path = hf_hub_download(
                    repo_id = hf_path,
                    filename=file_name,
                    force_download=self.force_download,
                    token=self.hf_token
                    )
            elif hf_path and (not file_name):
                if self.diffusers_model_check(hf_path):
                    #single_file = False
                    model_file_path = self._hf_repo_download(
                        url_or_path,
                        branch=branch
                        )
                else:
                    raise HTTPError("Invalid hf_path")
            else:
                raise TypeError("Invalid path_or_url")
        #from hf_repo
        elif self.diffusers_model_check(url_or_path):
            self.logger.debug(f"url_or_path: {url_or_path}")
            #single_file = False
            model_file_path = self._hf_repo_download(url_or_path,branch=branch)
        else:
            raise TypeError(f"Invalid path_or_url: {url_or_path}")
        return model_file_path # type: ignore


    def model_safe_check(self,model_list) ->str:
        if len(model_list)>1:
           for check_model in model_list:
                if bool(re.search(r"(?i)[-ー_＿](sfw|safe)", check_model)):
                    return check_model
        return model_list[0]


    def list_safe_sort(self,model_list) -> list:
        for check_model in model_list:
            if bool(re.search(r"(?i)[-ー_＿](sfw|safe)", check_model)):
                model_list.remove(check_model)
                model_list.insert(0, check_model)
                break
        return model_list


    def diffusers_model_check(
            self,
            checked_model: str,
            branch = "main"
            ) -> bool:
        index_url=f"https://huggingface.co/{checked_model}/blob/{branch}/model_index.json"
        return self.is_url_valid(index_url)


    def hf_model_check(self,path) -> bool:
        # Determine if a repository exists on the huggingface.
        return self.is_url_valid(f"https://huggingface.co/{path}")


    def model_data_get(
            self,
            path,
            model_info=None
            ) -> dict:
        
        data = model_info or self.hf_model_info(path)
        file_value_list = []
        df_model_bool=False
        # fix error': 'Repo model <repo_id>/<model> is gated. You must be authenticated to access it.
        try:
            siblings=data["siblings"]
        except KeyError:
            return {}

        for item in siblings:
            file_path=item["rfilename"]
            # model_index.json outside the root directory is not recognized
            if file_path=="model_index.json" and (not self.single_file_only):
                df_model_bool=True
            elif (
                any(file_path.endswith(ext) for ext in self.exts) and
                not any(file_path.endswith(ex) for ex in self.exclude)
                ):
                file_value_list.append(file_path)

        self.file_path_dict.update({path:(df_model_bool,file_value_list)})
        return {
            "model_info" : data,
            "file_list" : file_value_list,
            "security_risk" : self.hf_security_check(data)
            }


    def hf_model_search(
            self,
            model_path,
            limit_num
            ) -> list:
        params={
            "search" : model_path,
            "sort" : "likes",
            "direction" : -1,
            "limit" : limit_num,
            "fetch_config":True,
            "full":True
            }
        return [asdict(value) for value in list(self.hf_api.list_models(**params))]


    def old_hf_model_search(
            self,
            model_path,
            limit_num
            ):
        """
        NOTE:
        It is not already in use, but is kept as a spare.
        """
        url = f"https://huggingface.co/api/models"
        params = {
            "search" : model_path,
            "sort" : "likes",
            "direction" : -1,
            "limit" : limit_num
            }
        return requests.get(url,params=params).json()
    

    def hf_model_info(
            self,
            model_name
            ) -> dict:
        hf_info = self.hf_api.model_info(
            repo_id = model_name,
            token = self.hf_token,
            files_metadata=True,
            securityStatus = True
            )
        model_dict = asdict(hf_info)
        # When using asdict, securityStatus is not added to the dictionary and must be added separately.
        if "securityStatus" not in model_dict.keys():
            model_dict["securityStatus"] = hf_info.__dict__["securityStatus"]
        return model_dict
    

    def old_hf_model_info(self,model_select) -> dict:
        """
        NOTE:
        It is not already in use, but is kept as a spare.
        """
        url = f"https://huggingface.co/api/models/{model_select}"
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            raise HTTPError("A hugface login or token is required")
        data = response.json()
        return data
    

    def hf_security_check(self,check_dict):
        # Returns True if dangerous.
        # Also returns False in case of failure.
        try:
            if (check_dict["securityStatus"]["hasUnsafeFile"] or
                (not check_dict["securityStatus"]["scansDone"])
                ):
                return True
            else:
                return False
        except KeyError:
            return False
        

    def check_if_file_exists(self,hf_repo_info):
        try:
            return any(
                item['rfilename'].endswith(ext) for item in hf_repo_info['siblings'] for ext in self.exts
                )
        except KeyError:
            return False


    def hf_models(
            self,
            model_name,
            limit
            ) -> list:
        """
        return:
        repo_model_list,with_like : list
        """
        exclude_tag = ["audio-to-audio"]
        data = self.hf_model_search(
            model_name,
            limit
            )
        return_list = []
        for item in data:
            model_id = item["id"]
            like = item["likes"]
            private_value = item["private"]
            tag_value = item["tags"]
            file_list = self.get_hf_files(item)
            if (all(tag not in tag_value for tag in exclude_tag) and
                (not private_value)):
                #model_status = self.model_data_get(model_id)
                #if not model_status["security_risk"]:
                model_dict = {
                    "model_id":model_id,
                    "like" : like,
                    "model_info" : item,
                    "file_list" : file_list,
                    "security_risk" : False
                    }
                return_list.append(model_dict)
        if not return_list:
            print("No models matching your criteria were found on huggingface.")            
        return return_list
    

    def find_max_like(self,model_dict_list:list):
        """
        Finds the dictionary with the highest "like" value in a list of dictionaries.
        Args:
            model_dict_list: A list of dictionaries.

        Returns:
            The dictionary with the highest "like" value, or the first dictionary if none have "like".
        """
        max_like = 0
        max_like_dict = {}
        for model_dict in model_dict_list:
            if model_dict["like"] > max_like:
                max_like = model_dict["like"]
                max_like_dict = model_dict
        return max_like_dict["model_id"] or model_dict_list[0]["model_id"]
        

    def sort_by_likes(self,model_dict_list: list):
        return sorted(model_dict_list, key=lambda x: x.get("like", 0), reverse=True)
    

    def get_hf_files(self,check_data) -> list:
        check_file_value = []
        if check_data:
            siblings = check_data["siblings"]
            for item in siblings:
                fi_path=item["rfilename"]
                if (any(fi_path.endswith(ext) for ext in self.exts) and
                    (not any(fi_path.endswith(ex) for ex in self.exclude)) and
                    (not any(fi_path.endswith(st) for st in self.config_file_list))):
                    check_file_value.append(fi_path)
        return check_file_value
    
    
    def model_name_search(
            self,
            model_name:str,
            auto_set:bool,
            model_format:str = "single_file", # "all","diffusers"
            Recursive_execution:bool = False,
            extra_limit=None
            ):
        """
        auto_set: bool
        loads the model with the most likes in hugface

        About Parameters
            model_format(str): one of the following [“all”, “diffusers”, “single_file”].
        """
            
        if Recursive_execution:
            limit = 1000
        else:
            if extra_limit:
                limit = extra_limit
            else:
                limit = 15
        
        repo_model_list = self.hf_models(model_name,limit)
        previous_model_selection = self.check_func_hist(
            key="hf_model_name",
            return_value=True
            )
        models_to_exclude = self.check_func_hist(
            key="dangerous_model",
            return_value=True
            )
        if not auto_set:
            print("\033[34mThe following model paths were found\033[0m")
            if previous_model_selection is not None:
                print(f"Previous Choice: {previous_model_selection}")
            print("\033[34m0.Search civitai\033[0m")
            for (i,(model_dict)) in enumerate(repo_model_list,1):
                _hf_model_id = model_dict["model_id"]
                _hf_model_like = model_dict["like"]
                warning_txt = "\033[31m[danger]" if _hf_model_id in models_to_exclude else ""
                print(f"\033[34m{i}. {warning_txt}model path: {_hf_model_id}, evaluation: {_hf_model_like}\033[0m")

            if Recursive_execution:
                print("\033[34m16.Other than above\033[0m")

            while True:
                try:
                    choice = int(input("Select the model path to use: "))
                except ValueError:
                    print("\033[33mOnly natural numbers are valid.\033[0m")
                    continue
                if choice == 0:
                    return "_hf_no_model"
                elif (not Recursive_execution) and choice == len(repo_model_list)+1:
                    return self.model_name_search(
                        model_name = model_name,
                        auto_set = auto_set,
                        model_format=model_format,
                        Recursive_execution = True, 
                        extra_limit=extra_limit
                        )
                elif 1 <= choice <= len(repo_model_list):
                    choice_path_dict = repo_model_list[choice-1]
                    choice_path = choice_path_dict["model_id"]

                    # The process here excludes models that may have security problems.
                    if self.model_data_get(path=choice_path)["security_risk"]:
                        print("\033[31mThis model has a security problem.\033[0m")
                        continue
                    else:
                        break
                else:
                    print(f"\033[34mPlease enter the numbers 1~{len(repo_model_list)}\033[0m")
        else:
            if repo_model_list:
                for check_dict in self.sort_by_likes(repo_model_list):
                    repo_info = check_dict["model_info"]
                    check_repo = check_dict["model_id"]

                    # The process here excludes models that may have security problems.
                    if self.model_data_get(path=repo_info)["security_risk"]:
                        continue

                    if model_format == "diffusers" and self.diffusers_model_check(check_repo):
                        choice_path = check_repo
                        break
                    elif model_format == "single_file" and self.get_hf_files(repo_info):
                        choice_path = check_repo
                        break
                    elif model_format == "all" and (self.diffusers_model_check(check_repo) or self.get_hf_files(repo_info)):
                        choice_path = check_repo
                        break

                else:
                    if not Recursive_execution:
                        return self.model_name_search(
                                model_name = model_name,
                                auto_set = auto_set,
                                model_format = model_format,
                                Recursive_execution = True,
                                extra_limit=extra_limit
                                )
                    else:
                        self.logger.warning("No models in diffusers format were found.")
                        choice_path = "_hf_no_model"
            else:
                choice_path = "_hf_no_model"
                
        print("\033[0m",end="")#turn back the color
        return choice_path
    

    def file_name_set_sub(
            self,
            model_select,
            file_value
            ):
        check_key = f"{model_select}_select"
        if not file_value:
            if not self.diffuser_model:
                print("\033[31mNo candidates found at huggingface\033[0m")
                res = input("Searching for civitai?: ")
                if res.lower() in ["y","yes"]:
                    return "_hf_no_model"
                else:
                    raise ValueError("No available files were found in the specified repository")
            else:
                print("\033[34mOnly models in Diffusers format found\033[0m")
                while True:
                    result=input("Do you want to use it?[y/n]: ")
                    if result.lower() in ["y","yes"]:
                        return "_DFmodel"
                    elif result.lower() in ["n","no"]:
                        sec_result=input("Searching for civitai?[y/n]: ")
                        if sec_result.lower() in ["y","yes"]:
                            return "_hf_no_model"
                        elif sec_result.lower() in ["n","no"]:
                            raise ValueError("Processing was stopped because no corresponding model was found.")
                    else:
                        print("\033[34mPlease enter only [y,n]\033[0m")

        file_value=self.list_safe_sort(file_value)
        if len(file_value)>=self.num_prints: #15
            start_number="1"
            choice_history = self.check_func_hist(key = check_key,return_value=True)
            if choice_history:
                if choice_history>self.num_prints+1:
                    choice_history = self.num_prints+1
                print(f"\033[33m＊Previous number: {choice_history}\033[0m")

            if self.diffuser_model:
                start_number="0"
                print("\033[34m0.Use Diffusers format model")
            for i in range(self.num_prints):
                print(f"\033[34m{i+1}.File name: {file_value[i]}\033[0m")
            print(f"\033[34m{self.num_prints+1}.Other than the files listed above (all candidates will be displayed)\n")
            while True:
                choice = input(f"select the file you want to use({start_number}~21): ")
                try:
                    choice=int(choice)
                except ValueError:
                    print("\033[33mOnly natural numbers are valid\033[0m")
                    continue
                if self.diffuser_model and choice==0:
                    self.choice_number = -1
                    print("\033[0m",end="")
                    choice_history_update = self.check_func_hist(key=check_key,value=choice,update=True)
                    return "_DFmodel"
                
                elif choice==(self.num_prints+1): #other_file
                    break
                elif 1<=choice<=self.num_prints:
                    choice_path=file_value[choice-1]
                    self.choice_number = choice
                    choice_history_update = self.check_func_hist(key=check_key,value=choice,update=True)
                    return choice_path
                else:
                    print(f"\033[33mPlease enter numbers from 1~{self.num_prints}\033[0m")
            print("\n\n")

        choice_history = self.check_func_hist(key = check_key,return_value=True)
        if choice_history:
            print(f"\033[33m* Previous number: {choice_history}\033[0m")

        start_number="1"
        if self.diffuser_model:
            start_number="0"
            print("\033[34m0.Use Diffusers format model\033[0m")
        for i, file_name in enumerate(file_value, 1):
            print(f"\033[34m{i}.File name: {file_name}\033[0m")
        while True:
            choice = input(f"Select the file you want to use({start_number}~{len(file_value)}): ")
            try:
                choice=int(choice)
            except ValueError:
                print("\033[33mOnly natural numbers are valid\033[0m")
            else:
                if self.diffuser_model and choice==0:
                    self.choice_number = -1
                    choice_history_update = self.check_func_hist(key=check_key,value=choice,update=True)
                    return "_DFmodel"
                
                if 1<=choice<=len(file_value):
                    choice_path=file_value[choice-1]
                    self.choice_number = choice
                    choice_history_update = self.check_func_hist(key=check_key,value=choice,update=True)
                    return choice_path
                else:
                    print(f"\033[33mPlease enter numbers from 1~{len(file_value)}\033[34m")
                    

    def file_name_set(
            self,
            model_select,
            auto,
            model_format,
            model_type="Checkpoint",
            ):
        if self.diffusers_model_check(model_select) and model_type=="Checkpoint":
            self.diffuser_model=True
        
        if model_format == "single_file":
            skip_difusers = True
        else:
            skip_difusers = False

        data = self.hf_model_info(model_select)
        choice_path=""
        file_value = []
        if data:
            file_value = self.get_hf_files(check_data=data)
        else:
            raise ValueError("No available file was found.\nPlease check the name.")
        if file_value:
            file_value=self.sort_by_version(file_value)
            if not auto:
                print("\033[34mThe following model files were found\033[0m")
                choice_path=self.file_name_set_sub(model_select,file_value)
            else:
                if self.diffuser_model and (not skip_difusers):
                    choice_path = "_DFmodel"
                else:
                    choice_path=self.model_safe_check(file_value)

        elif self.diffuser_model:
            # When “auto” is selected, the presence or absence of “single_file” is determined when selecting a repo, so it is not necessary.
            choice_path = "_DFmodel"
        else:
            raise FileNotFoundError("No available files found in the specified repository")        
        return choice_path

