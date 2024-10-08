from dataclasses import dataclass

from omegaconf import MISSING

from jeffrey.config_schemas.base_schemas import TaskConfig
from jeffrey.config_schemas.evaluation import evaluation_lightning_module_schemas


@dataclass
class EvaluationTaskConfig(TaskConfig):
    pass


@dataclass
class TarModelEvaluationTaskConfig(EvaluationTaskConfig):
    tar_model_path: str = MISSING
    lightning_module: evaluation_lightning_module_schemas.PartialEvaluationLightningModuleConfig = MISSING
    

@dataclass
class CommonEvaluationTaskConfig(TarModelEvaluationTaskConfig):
    _target_: str = "jeffrey.evaluation.tasks.common_evaluation_task.CommonEvaluationTask"
    

@dataclass
class DefaultCommonEvaluationTaskConfig(CommonEvaluationTaskConfig):
    task_name: str = "binary_text_evaluation_task"
    lightning_module: evaluation_lightning_module_schemas.PartialEvaluationLightningModuleConfig = evaluation_lightning_module_schemas.BinaryTextEvaluationLightningModuleConfig()
    