import inspect
import diffusers
from natsort import natsorted

class basic_config:
    def __init__(self):
        pass
        
    
    def get_inherited_class(cls,class_name) -> list:
        inherited_class = inspect.getmro(class_name)
        return [cls_method.__name__ for cls_method in inherited_class]


    def get_item(self,dict_obj):
        """
        Returns the first element of the dictionary
        """
        return next(iter(dict_obj.items()))[1]


    def pipeline_metod_type(self,Target_class) -> str:
        """
        Args:
        Target_class : class

        Returns:
        Literal['torch','flax','onnx']
        """
        torch_list=["DiffusionPipeline",
                    "AutoPipelineForText2Image",
                    "AutoPipelineForImage2Image",
                    "AutoPipelineForInpainting",]

        flax_list = ["FlaxDiffusionPipeline",]

        if isinstance(Target_class,str):
            Target_class = getattr(diffusers, Target_class)

        cls_method= self.get_inherited_class(Target_class)

        if any(method in torch_list for method in cls_method):
            class_type= "torch"
        elif any(method in flax_list for method in cls_method):
            class_type= "flax"
        else:
            class_type= "onnx"
        return class_type


    def sort_by_version(self,sorted_list) -> list:
        """
        Returns:
        Sorted by version in order of newest to oldest
        """
        return natsorted(sorted_list,reverse = True)


    def key_check(self,keyword) -> bool:
        global key_dict
        if "key_dict" not in globals():
            key_dict = {}
        key = str(keyword)
        key_in = False
        if key in key_dict:
            if keyword == key_dict[key]:
                key_in = True
        key_dict[key] = keyword
        return key_in


    def get_call_method(
            self,
            class_name,
            method_name : str = '__call__'
            ) ->list:
        """
        Acquire the arguments of the function specified by 'method_name'
        for the class specified by 'class_name'
        """
        if isinstance(class_name,str):
            class_name = getattr(getattr(diffusers, class_name),method_name)
        parameters = inspect.signature(class_name).parameters
        arg_names = []
        for param in parameters.values():
            arg_names.append(param.name)
        return arg_names


    def get_class_elements(
            self,
            search
            ):
        return list(search.__class__.__annotations__.keys())
