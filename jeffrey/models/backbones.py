from torch import nn
from transformers import AutoConfig, AutoModel, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPooling

from jeffrey.models.transformations import Transformation
from jeffrey.utils.io_utils import translate_gcs_dir_to_local


class Backbone(nn.Module):
    def __init__(self, transformation: Transformation) -> None:
        super().__init__()
        self.transformation = transformation
        
    def get_transformation(self) -> Transformation:
        return self.transformation


class HuggingFaceBackbone(Backbone):
    def __init__(
        self, 
        pretrained_model_name_or_path: str, 
        transformation: Transformation, 
        pretrained: bool = False
    ) -> None:
        super().__init__(transformation=transformation)
        
        self.backbone = self.get_backbone(pretrained_model_name_or_path, pretrained)
        
    def forward(self, encodings: BatchEncoding) -> BaseModelOutputWithPooling:
        output = self.backbone(**encodings)
        return output
        
    def get_backbone(self, pretrained_model_name_or_path: str, pretrained: bool) -> nn.Module:
        path = translate_gcs_dir_to_local(pretrained_model_name_or_path)
        config = AutoConfig.from_pretrained(path)
        
        if pretrained:
            return AutoModel.from_pretrained(path, config=config)
        return AutoModel.from_config(config)
    