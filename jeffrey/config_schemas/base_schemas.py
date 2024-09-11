from dataclasses import dataclass
from omegaconf import MISSING

from jeffrey.config_schemas import data_module_schema
from jeffrey.config_schemas.trainer import trainer_schemas


@dataclass
class LightningModuleConfig:
    _target_: str = MISSING


@dataclass
class TaskConfig:
    _target_: str = MISSING
    task_name: str = MISSING
    data_module: data_module_schema.DataModuleConfig = MISSING
    lightning_module: LightningModuleConfig = MISSING
    trainer: trainer_schemas.TrainerConfig = MISSING