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
        self.path_dict = {
            "repo_id":"",
            "viersion_id":"",
            "filename":""
            }
        self.save_file_name = ""


    def civitai_download(
            self,
            seach_word,
            auto,
            model_type,
            download=True):
        """
        Function to download models from civitai.

        Parameters:
        - seach_word(str): Search query string.
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

        model_url, save_path = self.requests_civitai(
            query=seach_word,
            auto=auto,
            model_type=model_type)
        if model_url == "_civitai_no_model":
            return ("_civitai_no_model","_civitai_no_model")
        if download:
            self.download_model(
                url=model_url,
                save_path=save_path)
            return (model_url, self.civitai_save_path())
        else:
            return (model_url, model_url)


    def download_model(self, url, save_path):
        if not self.is_url_valid(url):
            raise requests.HTTPError("URL is invalid.")

        response = requests.get(url, stream=True)

        try:
            response.raise_for_status()
        except requests.HTTPError:
            raise requests.HTTPError(f"Invalid URL: {response.status_code}")

        os.makedirs(os.path.dirname(save_path),exist_ok=True)

        with tqdm.wrapattr(open(save_path, "wb"), "write",
            miniters=1, desc="Downloading model",
            total=int(response.headers.get('content-length', 0))) as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)
        self.logger.info(f"Downloaded file saved to {save_path}")


    def repo_select_civitai(self, state: list, auto: bool, recursive: bool = True):
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
            #self.logger.warning("There is no model in Civitai that fits the criteria.")
            #return "_civitai_no_model"
            raise ValueError("state is empty")

        if auto:
            return max(state, key=lambda x: x['downloadCount'])
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
            for number, states_dict in enumerate(sorted_list[:max_number]):
                print(f"\033[34m{number + 1}. Repo_id: {states_dict['CreatorName']} / {states_dict['repo_name']}, download: {states_dict['downloadCount']}")

            if Limit_choice:
                max_number += 1
                print(f"\033[34m{max_number}. Other than above")

            while True:
                try:
                    choice = int(input(f"choice repo [1~{max_number}]: "))
                except ValueError:
                    print("\033[33mOnly natural numbers are valid.\033[34m")
                    continue

                if Limit_choice and choice == max_number:
                    return self.repo_select_civitai(state=state, auto=auto, recursive=False)
                elif 1 <= choice <= max_number:
                    self.path_dict["repo_id"] = sorted_list[choice - 1]["repo_id"]
                    return sorted_list[choice - 1]
                else:
                    print(f"\033[33mPlease enter the numbers 1~{max_number}\033[34m")


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
            self.path_dict["version_id"] = result["id"]
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
                print(f"\033[34m{number_ + 1}. model_version: {state_dict_['name']}, download: {state_dict_['downloadCount']}")

            if Limit_choice:
                max_number += 1
                print(f"{max_number}. Other than above")


            while True:
                try:
                    choice = int(input("Select the model path to use: "))
                except ValueError:
                    print("\033[33mOnly natural numbers are valid.\033[34m")
                    continue
                if Limit_choice and choice == max_number:
                    return self.version_select_civitai(state=state, auto=auto, recursive=False)
                elif 1 <= choice <= max_number:
                    return_dict = ver_list[choice - 1]
                    self.path_dict["version_id"] = return_dict["id"]
                    return return_dict["files"]
                else:
                    print(f"\033[33mPlease enter the numbers 1~{max_number}\033[34m")


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
                print(f"{max_number}. Other than above")

            while True:
                try:
                    choice = int(input(f"Select the file to download[1~{max_number}]: "))
                except ValueError:
                    print("\033[33mOnly natural numbers are valid.\033[34m")
                    continue
                if Limit_choice and choice == max_number:
                    return self.file_select_civitai(state_list=state_list, auto=auto, recursive=False)
                elif 1 <= choice <= len(state_list):
                    self.path_dict["filename"] = state_list[choice - 1]["filename"]
                    return state_list[choice - 1]
                else:
                    print(f"\033[33mPlease enter the numbers 1~{len(state_list)}\033[34m")
        else:
            self.path_dict["filename"] = state_list[0]["filename"]
            return state_list[0]


    def civitai_save_path(self):
        """
        Set the save path using the information in path_dict.

        Returns:
        - str: Save path.
        """
        repo_level_dir = str(self.path_dict['repo_id'])
        file_version_dir = str(self.path_dict['version_id'])
        save_file_name = str(self.path_dict['filename'])
        save_path = os.path.join(self.base_civitai_dir, repo_level_dir, file_version_dir, save_file_name)
        return save_path


    def requests_civitai(self, query, auto, model_type):
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
        repo_list = []
        model_ver_list = []
        version_dict = {}

        params = {"query": query, "types": model_type, "sort": "Most Downloaded"}

        try:
            response = requests.get("https://civitai.com/api/v1/models", params=params)
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
                    if any(check_word in model_value for check_word in ["downloadUrl", "name"]):
                        file_status = {
                            "filename": model_value["name"],
                            "file_id": model_value["id"],
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
            return ("_civitai_no_model","_civitai_no_model")
            #raise ValueError("No matches found for your criteria")

        model_dict = self.repo_select_civitai(
            state = state,
            auto = auto
            )
        files_list = self.version_select_civitai(
            state = model_dict,
            auto = auto
            )

        file_status_dict = self.file_select_civitai(
            state_list = files_list,
            auto = auto)

        save_path = self.civitai_save_path()

        return (file_status_dict["download_url"], save_path)