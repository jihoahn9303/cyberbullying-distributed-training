from dataclasses import dataclass
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from jeffrey.config_schemas.models.transformations_schemas import (
    CustomHuggingFaceTokenizationTransformationConfig, 
    TransformationConfig
)
from jeffrey.utils.mixins import LoggerbleParamsMixin


@dataclass
class BackboneConfig(LoggerbleParamsMixin):
    _target_: str = MISSING
    transformation: TransformationConfig = MISSING
    
    def loggable_params(self) -> list[str]:
        return ["_target_"]
    

@dataclass
class HuggingFaceBackboneConfig(BackboneConfig):
    _target_: str = "jeffrey.models.backbones.HuggingFaceBackbone"
    pretrained_model_name_or_path: str = MISSING
    pretrained: bool = False
    
    def loggable_params(self) -> list[str]:
        return super().loggable_params() + ["pretrained_model_name_or_path", "pretrained"]
    
    
@dataclass
class BertTinyHuggingFaceBackboneConfig(HuggingFaceBackboneConfig):
    pretrained_model_name_or_path: str = "prajjwal1/bert-tiny"
    transformation: TransformationConfig = CustomHuggingFaceTokenizationTransformationConfig()
    

def register_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="huggingface_backbone_schema",
        group="tasks/lightning_module/model/backbone",
        node=HuggingFaceBackboneConfig
    )
    cs.store(
        name="test_backbone_config",
        node=BertTinyHuggingFaceBackboneConfig
    )