import os
import json


class config_check:
    base_config_json = "/tmp/auto_diffusers_config.json"
    def __init__(self):
        pass


    def check_func_hist(
            self,
            key,
            **kwargs
            ):
        """
        Check and optionally update the history of a given element.

        Args:
            key (str): Specific key to look up in the dictionary.
            **kwargs: Keyword arguments for additional options.
                - update (bool): Whether to update the dictionary. Default is True.
                - return_value (bool): Whether to return the element value. Default is False.
                - value (Any): Value to be matched or updated in the dictionary.
                - missing_value (Any): Returns the value if it does not exist.

        Returns:
            Any: The historical value if `return_value` is True, or a boolean indicating
                 if the value matches the historical value.
        """
        value = kwargs.pop("value",None)
        update = kwargs.pop("update", False if value is None else True)
        return_value = kwargs.pop("return_value", True if "value" in kwargs else False)
        missing_value = kwargs.pop("missing_value", None)

        hist_value = self.get_json_dict().get(key,None)
        if update:
            self.update_json_dict(key, value)

        if return_value:
            return hist_value or missing_value
        else:
            return hist_value == value


    def get_json_dict(self) -> dict:
        """
        Retrieve the JSON dictionary from the config file.
        """
        config_dict = {}
        if os.path.isfile(self.base_config_json):
            try:
                with open(self.base_config_json, "r") as basic_json:
                    config_dict = json.load(basic_json)
            except json.JSONDecodeError:
                pass
        return config_dict


    def update_json_dict(self, key, value):
        """Update the JSON dictionary with a new key-value pair."""
        basic_json_dict = self.get_json_dict()
        basic_json_dict[key] = value
        with open(self.base_config_json, "w") as json_file:
            json.dump(basic_json_dict, json_file, indent=4)


    