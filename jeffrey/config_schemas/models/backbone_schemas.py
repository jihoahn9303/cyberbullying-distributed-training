from dataclasses import dataclass
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class BackboneConfig:
    _target_: str = MISSING
    

@dataclass
class HuggingFaceBackboneConfig(BackboneConfig):
    _target_: str = "jeffrey.models.backbones.HuggingFaceBackbone"
    pretrained_model_name_or_path: str = MISSING
    pretrained: bool = False
    

def register_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="huggingface_backbone_schema",
        group="tasks/lightning_module/model/backbone",
        node=HuggingFaceBackboneConfig
    )