import os
import requests
from requests import HTTPError
from tqdm.auto import tqdm

from ..setup.base_config import Basic_config


class Civitai(Basic_config):
    '''
    Example:
    item = requests.get("http://civitai.example").json
    state_list = [{
        "repo_name": item["name"],
        "repo_id": item["id"],
        "favoriteCount": item["stats"]["favoriteCount"],
        "downloadCount": item["stats"]["downloadCount"],
        "CreatorName": item["creator"]["username"],
        "version_list": [{
            "id": item["modelVersions"]["id"],
            "name": item["modelVersions"]["name"],
            "downloadCount": item["modelVersions"]["downloadCount"],
            "files": [{
                "filename": item["modelVersions"]["files"]["name"],
                "file_id": item["modelVersions"]["files"]["id"],
                "download_url": item["modelVersions"]["files"]["downloadUrl"],
            }]
        }]
    }]
    return:
        state_list = {
            "repo_name": item["name"],
            "repo_id": item["id"],
            "favoriteCount": item["stats"]["favoriteCount"],
            "downloadCount": item["stats"]["downloadCount"],
            "CreatorName": item["creator"]["username"],
            "version_list": <file_list>
        }
    '''

    base_civitai_dir = "/root/.cache/Civitai"
    max_number_of_choices:int = 15
    chunk_size:int = 1024

    def __init__(self):
        super().__init__()
        self.save_file_name = ""
    

    def civitai_model_set(
            self,
            search_word,
            auto,
            model_type,
            download=True,
            civitai_token="",
            skip_error=True,
            include_hugface=True
            ):
        """
        Function to download models from civitai.

        Parameters:
        - search_word(str): Search query string.
        - auto(bool): Flag for automatic selection.
        - model_type(str): Type of model to search for.
            arg:[Checkpoint,
                 TextualInversion,
                 Hypernetwork,
                 AestheticGradient,
                 LORA,
                 Controlnet,
                 Poses
                ]
        - download(bool): Whether to download the model

        Returns:

        Local storage path if download is true,
        model download URL if false
        ---
        (model_url:str, save_path:str)
        ---
        """

        model_state_list = self.requests_civitai(
            query=search_word,
            auto=auto,
            model_type=model_type,
            civitai_token=civitai_token,
            include_hugface=include_hugface
            )
        
        if not model_state_list:
            if skip_error:
                return ""
            else:
                raise ValueError("No models were found in civitai.")
            
        model_url,model_save_path = model_state_list
        if download:    
            self.download_model(
                url=model_url,
                save_path=model_save_path,
                civitai_token=civitai_token
                )
            
            model_path = model_save_path
        else:
            model_path = model_url
        
        self.return_dict["model_status"]["single_file"] = True
        if download:
            self.return_dict["load_type"] = "from_single_file"
        else:
            self.return_dict["load_type"] = ""
 
        self.return_dict["model_path"] = model_path
        return model_path
    

    def civitai_security_check(self,value):
        """
        Note:
        The virus scan and pickle scan are used to make the decision.
        Returns True if judged to be dangerous.
        """
        check_list = [value["pickleScanResult"],value["virusScanResult"]]

        if all(status == "Success" for status in check_list):
            return False
        else:
            return True
    

    def requests_civitai(
            self, 
            query, 
            auto, 
            model_type,
            civitai_token="",
            include_hugface=True
            ):
        """
        Fetch models from Civitai based on a query and model type.

        Parameters:
        - query: Search query string.
        - auto: Flag for automatic selection.
        - model_type: Type of model to search for.
            arg:[Checkpoint,
                 TextualInversion,
                 Hypernetwork,
                 AestheticGradient,
                 LORA,
                 Controlnet,
                 Poses
                ]

        Returns:
        - str: Download URL of the selected file.
        (url, save_path)
        """
        state = []
        model_ver_list = []
        version_dict = {}

        params = {"query": query, "types": model_type, "sort": "Most Downloaded"}

        headers = {}
        if civitai_token:
            headers['Authorization'] = f'Bearer {civitai_token}'

        try:
            response = requests.get("https://civitai.com/api/v1/models", params=params, headers=headers)
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise HTTPError(f"Could not get elements from the URL. {err}")
        else:
            try:
                data = response.json()
            except AttributeError:
                raise ValueError("Invalid JSON response")

        items = data["items"]

        for item in items:
            for model_ver in item["modelVersions"]:
                files_list = []
                for model_value in model_ver["files"]:
                    security_risk = self.civitai_security_check(model_value)
                    if (any(check_word in model_value for check_word in ["downloadUrl", "name"]) and
                        not security_risk
                        ):
                        file_status = {
                            "filename": model_value["name"],
                            "file_id": model_value["id"],
                            "fp": model_value["metadata"]["fp"],
                            "file_format": model_value["metadata"]["format"],                
                            "download_url": model_value["downloadUrl"],
                        }
                        files_list.append(file_status)

                version_dict = {
                    "id": model_ver["id"],
                    "name": model_ver["name"],
                    "downloadCount": model_ver["stats"]["downloadCount"],
                    "files": files_list,
                }

                if files_list:
                    model_ver_list.append(version_dict)

            if all(check_txt in item.keys() for check_txt in ["name", "stats", "creator"]):
                state_dict = {
                    "repo_name": item["name"],
                    "repo_id": item["id"],
                    "favoriteCount": item["stats"]["favoriteCount"],
                    "downloadCount": item["stats"]["downloadCount"],
                    "CreatorName": item["creator"]["username"],
                    "version_list": model_ver_list,
                }

                if model_ver_list:
                    state.append(state_dict)

        if not state:
            self.logger.warning("There is no model in Civitai that fits the criteria.")
            return {}
            #raise ValueError("No matches found for your criteria")

        dict_of_civitai_repo = self.repo_select_civitai(
            state = state,
            auto = auto,
            include_hugface=include_hugface
            )
        
        if not dict_of_civitai_repo:
            return []
        
        files_list = self.version_select_civitai(
            state = dict_of_civitai_repo,
            auto = auto
            )

        file_status_dict = self.file_select_civitai(
            state_list = files_list,
            auto = auto
            )

        save_path = self.civitai_save_path()
        return [file_status_dict["download_url"],save_path] 


    def repo_select_civitai(
            self,
            state: list, 
            auto: bool, 
            recursive: bool = True,
            include_hugface: bool = True):
        """
        Set repository requests for Civitai.

        Parameters:
        - state (list): List of repository information.
        - auto (bool): Flag for automatic selection.
        - recursive (bool): Flag for recursion.

        Returns:
        - dict: Selected repository information.
        """
        if not state:
            self.logger.warning("No models were found in civitai.")
            return {}

        elif auto:
            repo_dict = max(state, key=lambda x: x['downloadCount'])
            self.return_dict["repo_status"]["repo_name"] = repo_dict["repo_name"]
            self.return_dict["repo_status"]["repo_id"] = repo_dict["repo_id"]
            return repo_dict
        else:
            sorted_list = sorted(state, key=lambda x: x['downloadCount'], reverse=True)
            if recursive and self.max_number_of_choices < len(sorted_list):
                Limit_choice = True
            else:
                Limit_choice = False

            if recursive:
                print("\n\n\033[34mThe following repo paths were found\033[0m")
            else:
                print("\n\n\n")

            max_number = min(self.max_number_of_choices, len(sorted_list)) if recursive else len(sorted_list)
            if include_hugface:
                print(f"\033[34m0. Search for huggingface\033[0m")
            for number, states_dict in enumerate(sorted_list[:max_number]):
                print(f"\033[34m{number + 1}. Repo_id: {states_dict['CreatorName']} / {states_dict['repo_name']}, download: {states_dict['downloadCount']}\033[0m")

            if Limit_choice:
                max_number += 1
                print(f"\033[34m{max_number}. Other than above\033[0m")

            while True:
                try:
                    choice = int(input(f"choice repo [1~{max_number}]: "))
                except ValueError:
                    print("\033[33mOnly natural numbers are valid.\033[0m")
                    continue

                if Limit_choice and choice == max_number:
                    return self.repo_select_civitai(state=state, auto=auto, recursive=False)
                elif choice == 0 and include_hugface:
                    return {}
                elif 1 <= choice <= max_number:
                    repo_dict = sorted_list[choice - 1]
                    self.return_dict["repo_status"]["repo_name"] = repo_dict["repo_name"]
                    self.return_dict["repo_status"]["repo_id"] = repo_dict["repo_id"]
                    return repo_dict
                else:
                    print(f"\033[33mPlease enter the numbers 1~{max_number}\033[0m")
                

    def download_model(
            self, 
            url, 
            save_path, 
            civitai_token=""
            ):
        if not self.is_url_valid(url):
            raise requests.HTTPError("URL is invalid.")
        
        headers = {}
        if civitai_token:
            headers['Authorization'] = f'Bearer {civitai_token}'

        response = requests.get(url, stream=True, headers=headers)

        try:
            response.raise_for_status()
        except requests.HTTPError:
            raise requests.HTTPError(f"Invalid URL: {response.status_code}")

        os.makedirs(os.path.dirname(save_path),exist_ok=True)

        with tqdm.wrapattr(open(save_path, "wb"), "write",
            miniters=1, desc="Downloading model",
            total=int(response.headers.get('content-length', 0))) as fout:
            for chunk in response.iter_content(chunk_size=8192):
                fout.write(chunk)
        self.logger.info(f"Downloaded file saved to {save_path}")


    def version_select_civitai(self, state, auto, recursive: bool = True):
        """
        Set model requests for Civitai.

        Parameters:
        - state: Model information state.
        - auto: Flag for automatic selection.
        - recursive (bool): Flag for recursion.

        Returns:
        - dict: Selected model information.
        """
        if not state:
            raise ValueError("state is empty")

        ver_list = sorted(state["version_list"], key=lambda x: x['downloadCount'], reverse=True)

        if recursive and self.max_number_of_choices < len(ver_list):
            Limit_choice = True
        else:
            Limit_choice = False

        if auto:
            result = max(ver_list, key=lambda x: x['downloadCount'])
            ver_files_list = self.sort_by_version(result["files"])
            self.return_dict["repo_status"]["version_id"] = result["id"]
            return ver_files_list
        else:
            if recursive:
                print("\n\n\033[34mThe following model paths were found\033[0m")
            else:
                print("\n\n\n")

            if len(ver_list) == 1:
                return ver_list

            max_number = min(self.max_number_of_choices, len(ver_list)) if recursive else len(ver_list)

            for number_, state_dict_ in enumerate(ver_list[:max_number]):
                print(f"\033[34m{number_ + 1}. model_version: {state_dict_['name']}, download: {state_dict_['downloadCount']}\033[0m")

            if Limit_choice:
                max_number += 1
                print(f"\033[34m{max_number}. Other than above\033[0m")


            while True:
                try:
                    choice = int(input("Select the model path to use: "))
                except ValueError:
                    print("\033[33mOnly natural numbers are valid.\033[0m")
                    continue
                if Limit_choice and choice == max_number:
                    return self.version_select_civitai(state=state, auto=auto, recursive=False)
                elif 1 <= choice <= max_number:
                    return_dict = ver_list[choice - 1]
                    self.return_dict["repo_status"]["version_id"] = return_dict["id"]
                    return return_dict["files"]
                else:
                    print(f"\033[33mPlease enter the numbers 1~{max_number}\033[0m")


    def file_select_civitai(self, state_list, auto,recursive:bool=True):
        """
        Return the download URL for the selected file.

        Parameters:
        - state_list: List of file information.
        - auto: Flag for automatic selection.

        Returns:
        - str: Download URL of the selected file.
        """
        
        if recursive and self.max_number_of_choices < len(state_list):
            Limit_choice = True
        else:
            Limit_choice = False

        if len(state_list) > 1 and (not auto):
            max_number = min(self.max_number_of_choices, len(state_list)) if recursive else len(state_list)
            for number, states_dict in enumerate(state_list[:max_number]):
                print(f"\033[34m{number + 1}. File_name: {states_dict['filename']}")

            if Limit_choice:
                max_number += 1
                print(f"\033[34m{max_number}. Other than above\033[0m")

            while True:
                try:
                    choice = int(input(f"Select the file to download[1~{max_number}]: "))
                except ValueError:
                    print("\033[33mOnly natural numbers are valid.\033[0m")
                    continue
                if Limit_choice and choice == max_number:
                    return self.file_select_civitai(state_list=state_list, auto=auto, recursive=False)
                elif 1 <= choice <= len(state_list):
                    #self.return_dict.update(state_list[choice - 1])
                    file_dict = state_list[choice - 1]
                    self.return_dict["model_status"].update(file_dict)
                    return file_dict
                else:
                    print(f"\033[33mPlease enter the numbers 1~{len(state_list)}\033[0m")
        else:
            file_dict = state_list[0]
            self.return_dict["model_status"].update(file_dict)
            return state_list[0]


    def civitai_save_path(self):
        """
        Set the save path using the information in path_dict.

        Returns:
        - str: Save path.
        """
        repo_level_dir = str(self.return_dict["repo_status"]["repo_id"])
        file_version_dir = str(self.return_dict["repo_status"]["version_id"])
        save_file_name = str(self.return_dict["model_status"]["filename"])
        save_path = os.path.join(self.base_civitai_dir, repo_level_dir, file_version_dir, save_file_name)
        self.return_dict["model_path"] = save_path
        return save_path