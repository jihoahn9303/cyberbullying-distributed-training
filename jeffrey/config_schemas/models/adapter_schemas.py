from dataclasses import dataclass, field
from typing import List, Literal, Optional
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class AdapterConfig:
    _target_: str = MISSING
    

@dataclass
class MLPWithPoolingConfig(AdapterConfig):
    _target_: str = "jeffrey.models.adapters.MLPWithPooling"
    output_feature_sizes: List[int] = MISSING
    biases: Optional[List[bool]] = None
    activation_funcs: Optional[List[str]] = None
    dropout_rates: Optional[List[float]] = None
    batch_norms: Optional[List[bool]] = None
    order: str = "LBADN"
    standardize_input: bool = True
    pooling_method: Optional[str] = None
    output_attribute_to_use: Optional[str] = None
    
    
@dataclass
class PoolerOutputAdapterConfig(MLPWithPoolingConfig):
    output_feature_sizes: List[int] = field(default_factory=lambda: [-1])
    output_attribute_to_use: Optional[str] = "pooler_output"
    

def register_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="mlp_with_pooling_schema",
        group="tasks/lightning_module/model/adapter",
        node=MLPWithPoolingConfig
    )