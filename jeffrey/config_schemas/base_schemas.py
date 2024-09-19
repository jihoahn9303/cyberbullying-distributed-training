from dataclasses import dataclass
from omegaconf import MISSING

from jeffrey.config_schemas import data_module_schema
from jeffrey.config_schemas.trainer import trainer_schemas
from jeffrey.utils.mixins import LoggerbleParamsMixin


@dataclass
class LightningModuleConfig(LoggerbleParamsMixin):
    _target_: str = MISSING


@dataclass
class TaskConfig(LoggerbleParamsMixin):
    _target_: str = MISSING
    task_name: str = MISSING
    data_module: data_module_schema.DataModuleConfig = MISSING
    lightning_module: LightningModuleConfig = MISSING
    trainer: trainer_schemas.TrainerConfig = MISSING
    
    def loggable_params(self) -> list[str]:
        return ["_target_"]