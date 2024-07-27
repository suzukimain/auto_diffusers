class config_check:
    base_config_json = "/tmp/auto_diffusers_config.json"
    def __init__(self):
        pass


    def get_json_dict(self):
        """Retrieve the JSON dictionary from the config file."""
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


    def check_func_hist(self,*return_key, **kwargs):
        """
        Check and optionally update the history of a given element.

        Args:
            *return_key (str): Variable for which to get the history.
            **kwargs: Keyword arguments for additional options.
                - update (bool): Whether to update the dictionary. Default is True.
                - return_value (bool): Whether to return the element value. Default is False.
                - key (str): Specific key to look up in the dictionary.
                - value (Any): Value to be matched or updated in the dictionary.

        Returns:
            Any: The historical value if `return_value` is True, or a boolean indicating
                 if the value matches the historical value.
        """
        update = kwargs.pop("update", True)
        return_value = kwargs.pop("return_value", False)
        if kwargs:
            if "key" in kwargs:
                key = kwargs["key"]
                if "value" in kwargs:
                    value = kwargs["value"]
                else:
                    value = None
                    update = False
                    return_value = True
            else:
                key, value = next(iter(kwargs.items()))
        elif return_key:
            key, value = return_key[0], None
            update = False
            return_value = True
        else:
            raise TypeError("Missing 'key' argument.")
          
        basic_json_dict = self.get_json_dict()
        hist_value = basic_json_dict.get(key)
        if hist_value == value:
            value_match = True
        else:
            value_match = False

        if update:
            self.update_json_dict(key, value)

        if return_value:
            return hist_value
        else:
            return value_match
