from abc import ABC, abstractmethod
from typing import Union, TYPE_CHECKING

from lightning.pytorch import Trainer

from jeffrey.data_modules.data_modules import DataModule, PartialDataModule
from jeffrey.evaluation.lightning_modules.bases import EvaluationLightningModule, PartialEvaluationLightningModuleType
from jeffrey.models.common.exporter import TarModelLoader

if TYPE_CHECKING:
    from jeffrey.config_schemas.config_schema import Config
    from jeffrey.config_schemas.evaluation.evaluation_task_schemas import EvaluationTaskConfig


class EvaluationTask(ABC):
    def __init__(
        self,
        task_name: str,
        data_module: Union[DataModule, PartialDataModule],
        lightning_module: EvaluationLightningModule,
        trainer: Trainer
    ) -> None:
        super().__init__()
        
        self.task_name = task_name
        self.trainer = trainer
        self.lightning_module = lightning_module
        self.lightning_module.eval()
        
        if isinstance(data_module, DataModule):
            self.data_module = data_module
        else:
            self.data_module = data_module(transformation=self.lightning_module.get_transformation())
            
    @abstractmethod
    def run(self, config: "Config", task_config: "EvaluationTaskConfig") -> None:
        ...
        

class TarModelEvaluationTask(EvaluationTask):
    def __init__(
        self, 
        task_name: str, 
        data_module: Union[DataModule, PartialDataModule], 
        lightning_module: PartialEvaluationLightningModuleType, 
        trainer: Trainer,
        tar_model_path: str,
    ) -> None:
        model = TarModelLoader(exported_model_path=tar_model_path).load()
        _lightning_module = lightning_module(model=model)
        
        super().__init__(
            task_name=task_name, 
            data_module=data_module, 
            lightning_module=_lightning_module, 
            trainer=trainer
        )