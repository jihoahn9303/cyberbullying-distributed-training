from dataclasses import dataclass
from omegaconf import SI
from hydra.core.config_store import ConfigStore

from jeffrey.config_schemas.base_schemas import TaskConfig
from jeffrey.config_schemas import data_module_schema
from jeffrey.config_schemas.training import lightning_module_schemas
from jeffrey.config_schemas.trainer import trainer_schemas


@dataclass
class TrainingTaskConfig(TaskConfig):
    best_training_checkpoint: str = SI("${infrastructure.mlflow.artifact_uri}/best-checkpoints/")
    last_training_checkpoint: str = SI("${infrastructure.mlflow.artifact_uri}/last-checkpoints/")
    

@dataclass
class CommonTrainingTaskConfig(TrainingTaskConfig):
    _target_: str = "jeffrey.training.tasks.common_training_task.CommonTrainingTask"
    

@dataclass
class DefaultCommonTrainingTaskConfig(CommonTrainingTaskConfig):
    task_name: str = "binary_text_classification_task"
    data_module: data_module_schema.DataModuleConfig = data_module_schema.ScrappedDataTextClassificationDataModuleConfig()
    lightning_module: lightning_module_schemas.TrainingLightningModuleConfig = lightning_module_schemas.DefaultBinaryTextClassificationTrainingLightningModuleConfig()
    trainer: trainer_schemas.TrainerConfig = trainer_schemas.GPUDevConfig()


def register_config() -> None:
    data_module_schema.register_config()
    lightning_module_schemas.register_config()
    trainer_schemas.register_config()
    
    cs = ConfigStore.instance()
    cs.store(
        name="common_training_task_schema",
        group="tasks",
        node=CommonTrainingTaskConfig
    )
    cs.store(
        name="test_training_task_schema",
        node=DefaultCommonTrainingTaskConfig
    )