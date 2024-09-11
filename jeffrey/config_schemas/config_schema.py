from typing import Dict
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass

from jeffrey.config_schemas.infrastructure import infrastructure_schema
from jeffrey.config_schemas.training import training_task_schemas
from jeffrey.config_schemas import base_schemas


@dataclass
class Config:
    infrastructure: infrastructure_schema.InfrastructureConfig = infrastructure_schema.InfrastructureConfig()
    save_last_checkpoint_every_n_train_steps: int = 50
    seed: int = 1234
    tasks: Dict[str, base_schemas.TaskConfig] = MISSING


def register_config() -> None:
    infrastructure_schema.register_config()
    training_task_schemas.register_config()
    
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
