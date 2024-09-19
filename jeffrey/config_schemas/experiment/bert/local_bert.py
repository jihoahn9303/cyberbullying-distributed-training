from dataclasses import dataclass, field
from typing import Dict
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

from jeffrey.config_schemas.base_schemas import TaskConfig
from jeffrey.config_schemas.config_schema import Config
from jeffrey.config_schemas.evaluation.evaluation_task_schemas import DefaultCommonEvaluationTaskConfig
from jeffrey.config_schemas.training.training_task_schemas import DefaultCommonTrainingTaskConfig


@dataclass
class LocalBertExperiment(Config):
    tasks: Dict[str, TaskConfig] = field(
        default_factory=lambda: {
            'binary_text_classification_task': DefaultCommonTrainingTaskConfig(),
            'binary_text_evaluation_task': DefaultCommonEvaluationTaskConfig()
        }
    )
    
FinalLocalBertExperiment = OmegaConf.merge(
    LocalBertExperiment,
    OmegaConf.from_dotlist(
        [
            "tasks.binary_text_evaluation_task.tar_model_path=${tasks.binary_text_classification_task.tar_model_export_path}",
            "tasks.binary_text_evaluation_task.data_module=${tasks.binary_text_classification_task.data_module}",
            "tasks.binary_text_evaluation_task.trainer=${tasks.binary_text_classification_task.trainer}"
        ]
    )
)

cs = ConfigStore.instance()
cs.store(
    name='local_bert',
    group='experiment/bert',
    node=FinalLocalBertExperiment,
    package="_global_"
)