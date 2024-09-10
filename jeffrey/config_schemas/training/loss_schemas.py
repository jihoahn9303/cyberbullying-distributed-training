from dataclasses import dataclass
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class LossFunctionConfig:
    _target_: str = MISSING
    
    
@dataclass
class BCEWithLogitsLossConfig:
    _target_: str = "jeffrey.training.loss_functions.BCEWithLogitsLoss"
    reduction: str = "mean"
    

def register_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="bce_with_logits_loss_schema",
        group="tasks/lightning_module/loss",
        node=BCEWithLogitsLossConfig
    )