from typing import TYPE_CHECKING, Union

from lightning import Trainer

from jeffrey.data_modules.data_modules import DataModule, PartialDataModule
from jeffrey.models.common.exporter import TarModelExporter
from jeffrey.training.lightning_modules.bases import ModelStateDictExportingTrainingLightningModule
from jeffrey.training.tasks.bases import TrainingTask
from jeffrey.utils.io_utils import is_file
from jeffrey.utils.mlflow_utils import activate_mlflow, log_artifacts_for_reproducibility

if TYPE_CHECKING:
    from jeffrey.config_schemas.config_schema import Config 
    from jeffrey.config_schemas.training.training_task_schemas import TrainingTaskConfig


class TarModelExportingTrainingTask(TrainingTask):
    def __init__(
        self,
        task_name: str,
        data_module: Union[DataModule, PartialDataModule],
        lightning_module: ModelStateDictExportingTrainingLightningModule,
        trainer: Trainer,
        best_training_checkpoint: str,
        last_training_checkpoint: str,
        tar_model_export_path: str
    ) -> None:
        super().__init__(
            task_name=task_name,
            data_module=data_module,
            lightning_module=lightning_module,
            trainer=trainer,
            best_training_checkpoint=best_training_checkpoint,
            last_training_checkpoint=last_training_checkpoint
        )
        
        self.tar_model_export_path = tar_model_export_path
        
    def run(self, config: "Config", task_config: "TrainingTaskConfig") -> None:
        experiment_name = config.infrastructure.mlflow.experiment_name
        run_id = config.infrastructure.mlflow.run_id
        run_name = config.infrastructure.mlflow.run_name
        
        with activate_mlflow(
            experiment_name=experiment_name,
            run_id=run_id,
            run_name=run_name
        ) as _:
            if self.trainer.is_global_zero:
                log_artifacts_for_reproducibility()
                # log_training_hparams(config)
            
            if is_file(self.last_training_checkpoint):
                self.logger.info(
                    """
                    Found checkpoint here: {self.last_training_checkpoint}.
                    Resuming trainig...
                    """
                )
                assert isinstance(self.data_module, DataModule)
                self.trainer.fit(
                    model=self.lightning_module, 
                    datamodule=self.data_module, 
                    ckpt_path=self.last_training_checkpoint
                )
            else:
                self.trainer.fit(model=self.lightning_module, datamodule=self.data_module)
                
            self.logger.info("Training finished!!")
            self.logger.info("Exporting model state dict...")
            
            model_state_dict_path = self.lightning_module.export_model_state_dict(self.best_training_checkpoint)
            model_config = task_config.lightning_module.model
            model_exporter = TarModelExporter(
                model_state_dict_path=model_state_dict_path,
                model_config=model_config,
                tar_model_export_path=self.tar_model_export_path
            )
            model_exporter.export()