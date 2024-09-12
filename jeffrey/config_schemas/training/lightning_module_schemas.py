from dataclasses import dataclass
from typing import Optional
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from jeffrey.config_schemas.base_schemas import LightningModuleConfig
from jeffrey.config_schemas.models import model_schemas
from jeffrey.config_schemas.training import (
    loss_schemas,
    optimizer_schemas,
    scheduler_schemas
)

@dataclass
class TrainingLightningModuleConfig(LightningModuleConfig):
    _target_: str = MISSING
    model: model_schemas.ModelConfig = MISSING
    loss: loss_schemas.LossFunctionConfig = MISSING
    optimizer: optimizer_schemas.OptimizerConfig = MISSING
    scheduler: Optional[scheduler_schemas.LightningSchedulerConfig] = None
    

@dataclass
class BinaryTextClassificationTrainingLightningModuleConfig(TrainingLightningModuleConfig):
    _target_: str = "jeffrey.training.lightning_modules.binary_text_classification.BinaryTextClassificationLightningModule"
    
    
@dataclass
class DefaultBinaryTextClassificationTrainingLightningModuleConfig(BinaryTextClassificationTrainingLightningModuleConfig):
    model: model_schemas.ModelConfig = model_schemas.BertTinyBinaryTextClassificationModelConfig()
    loss: loss_schemas.LossFunctionConfig = loss_schemas.BCEWithLogitsLossConfig()
    optimizer: optimizer_schemas.OptimizerConfig = optimizer_schemas.AdamWOptimizerConfig()
    scheduler: Optional[scheduler_schemas.LightningSchedulerConfig] = scheduler_schemas.ReduceLROnPlateauLightningSchedulerConfig()


def register_config() -> None:
    # model_schemas.register_config()
    # loss_schemas.register_config()
    # optimizer_schemas.register_config()
    # scheduler_schemas.register_config()
    
    cs = ConfigStore.instance()
    cs.store(
        name="binary_text_classification_training_lightning_module_schema",
        group="tasks/lightning_module",
        node=BinaryTextClassificationTrainingLightningModuleConfig
    )
    
    