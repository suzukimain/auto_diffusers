class Pipeline_status:
    def pipe_class_type(
            self,
            class_name
            ):
        """
        Args:
        class_name : class

        Returns:
        Literal['txt2img','img2img','txt2video']
        """
        _txt2img_method_list = [] #else
        _img2img_method_list = ["image"]
        _img2video_method_list = ["video_length","fps"]

        call_method = self.get_call_method(class_name,method_name = '__call__')

        if any(method in call_method for method in _img2video_method_list):
            pipeline_type = "txt2video"
        elif any(method in call_method for method in _img2img_method_list):
            pipeline_type = "img2img"
        else:
            pipeline_type = "txt2img"
        return pipeline_type


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
