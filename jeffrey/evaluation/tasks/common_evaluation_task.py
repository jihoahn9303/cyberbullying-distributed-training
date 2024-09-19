from typing import TYPE_CHECKING, Union

from lightning import Trainer

from jeffrey.data_modules.data_modules import DataModule, PartialDataModule
from jeffrey.evaluation.lightning_modules.bases import PartialEvaluationLightningModuleType
from jeffrey.evaluation.tasks.bases import TarModelEvaluationTask
from jeffrey.utils.mlflow_utils import activate_mlflow

if TYPE_CHECKING:
    from jeffrey.config_schemas.config_schema import Config 
    from jeffrey.config_schemas.evaluation.evaluation_task_schemas import EvaluationTaskConfig


class CommonEvaluationTask(TarModelEvaluationTask):
    def __init__(
        self,
        task_name: str,
        data_module: Union[DataModule, PartialDataModule],
        lightning_module: PartialEvaluationLightningModuleType,
        trainer: Trainer,
        tar_model_path: str
    ) -> None:
        super().__init__(
            task_name=task_name,
            data_module=data_module,
            lightning_module=lightning_module,
            trainer=trainer,
            tar_model_path=tar_model_path
        )
        
    def run(self, config: "Config", task_config: "EvaluationTaskConfig") -> None:
        experiment_name = config.infrastructure.mlflow.experiment_name
        run_id = config.infrastructure.mlflow.run_id
        run_name = config.infrastructure.mlflow.run_name
        
        with activate_mlflow(
            experiment_name=experiment_name,
            run_id=run_id,
            run_name=run_name
        ) as _:
            self.trainer.test(model=self.lightning_module, datamodule=self.data_module)