from dataclasses import dataclass, field
from typing import Dict, Optional
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

from jeffrey.config_schemas.base_schemas import TaskConfig
from jeffrey.config_schemas.config_schema import Config
from jeffrey.config_schemas.evaluation import model_selector_schemas
from jeffrey.config_schemas.evaluation.evaluation_task_schemas import DefaultCommonEvaluationTaskConfig
from jeffrey.config_schemas.trainer.trainer_schemas import GPUProdConfig
from jeffrey.config_schemas.training.training_task_schemas import DefaultCommonTrainingTaskConfig


@dataclass
class LocalBertExperiment(Config):
    tasks: Dict[str, TaskConfig] = field(
        default_factory=lambda: {
            'binary_text_classification_task': DefaultCommonTrainingTaskConfig(trainer=GPUProdConfig()),
            'binary_text_evaluation_task': DefaultCommonEvaluationTaskConfig()
        }
    )
    model_selector: Optional[model_selector_schemas.ModelSelectorConfig] = model_selector_schemas.CyberBullyingDetectionModelSelectorConfig()
    registered_model_name: Optional[str] = "bert_tiny"
    
FinalLocalBertExperiment = OmegaConf.merge(
    LocalBertExperiment,
    OmegaConf.from_dotlist(
        [
            "infrastructure.mlflow.experiment_name=cyberbullying-detection",
            "tasks.binary_text_classification_task.data_module.batch_size=512",
            "tasks.binary_text_classification_task.data_module.transformation.max_sequence_len=150",
            "tasks.binary_text_classification_task.lightning_module.optimizer.lr=3e-6",
            "tasks.binary_text_classification_task.lightning_module.optimizer.weight_decay=1e-2",
            "tasks.binary_text_classification_task.trainer.max_epochs=20",
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