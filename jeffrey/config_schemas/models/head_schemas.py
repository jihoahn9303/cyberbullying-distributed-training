from dataclasses import dataclass
from typing import List, Literal, Optional
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class HeadConfig:
    _target_: str = MISSING
    

@dataclass
class SigmoidHeadConfig(HeadConfig):
    _target_: str = "jeffrey.models.heads.SigmoidHead"
    in_features: int = MISSING
    out_features: int = MISSING
    

def register_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="sigmoid_head_schema",
        group="tasks/lightning_module/model/head",
        node=SigmoidHeadConfig
    )