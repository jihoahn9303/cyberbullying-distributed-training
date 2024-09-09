from dataclasses import dataclass
from typing import Optional
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from jeffrey.config_schemas.models import backbone_schemas, adapter_schemas, head_schemas


@dataclass
class ModelConfig:
    _target_: str = MISSING
    

@dataclass
class BinaryTextClassificationModelConfig(ModelConfig):
    _target_: str = "jeffrey.models.models.BinaryTextClassificationModel"
    backbone: backbone_schemas.BackboneConfig = MISSING
    head: head_schemas.HeadConfig = MISSING
    adapter: Optional[adapter_schemas.AdapterConfig] = None
    

def register_config() -> None:
    backbone_schemas.register_config()
    adapter_schemas.register_config
    head_schemas.register_config()
    
    cs = ConfigStore.instance()
    cs.store(
        name="binary_text_classfication_model_schema",
        group="tasks/lightning_module/model",
        node=BinaryTextClassificationModelConfig
    )