from dataclasses import dataclass
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from jeffrey.utils.mixins import LoggerbleParamsMixin


@dataclass
class LossFunctionConfig(LoggerbleParamsMixin):
    _target_: str = MISSING
    
    def loggable_params(self) -> list[str]:
        return ["_target_"]
    
    
@dataclass
class BCEWithLogitsLossConfig(LossFunctionConfig):
    _target_: str = "jeffrey.training.loss_functions.BCEWithLogitsLoss"
    reduction: str = "mean"
    

def register_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="bce_with_logits_loss_schema",
        group="tasks/lightning_module/loss",
        node=BCEWithLogitsLossConfig
    )