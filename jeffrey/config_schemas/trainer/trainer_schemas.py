from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore

from jeffrey.config_schemas.trainer import logger_schemas, callbacks_schemas
from jeffrey.utils.mixins import LoggerbleParamsMixin


@dataclass
class TrainerConfig(LoggerbleParamsMixin):
    _target_: str = "lightning.pytorch.trainer.trainer.Trainer"
    accelerator: str = "auto"
    strategy: str = "ddp_find_unused_parameters_true"
    devices: str = "auto"
    num_nodes: int = 1  # SI("${infrastructure.vm_comfig.node_count}")  
    precision: str = "16-mixed"
    logger: Optional[list[logger_schemas.LoggerConfig]] = field(default_factory=lambda: [])  # type: ignore
    callbacks: Optional[list[callbacks_schemas.CallbackConfig]] = field(default_factory=lambda: [])  # type: ignore
    fast_dev_run: bool = False
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: int = -1
    min_steps: Optional[int] = None
    max_time: Optional[str] = None
    limit_train_batches: Optional[float] = 1
    limit_val_batches: Optional[float] = 1
    limit_test_batches: Optional[float] = 1
    limit_predict_batches: Optional[float] = 1
    overfit_batches: float = 0.0
    val_check_interval: Optional[float] = 1
    check_val_every_n_epoch: Optional[int] = 1
    num_sanity_val_steps: int = 2
    log_every_n_steps: int = 50
    enable_checkpointing: bool = True
    enable_progress_bar: bool = True
    enable_model_summary: bool = True
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[float] = 5
    gradient_clip_algorithm: Optional[str] = "value"
    deterministic: Optional[bool] = None
    benchmark: Optional[bool] = None
    inference_mode: bool = True
    use_distributed_sampler: bool = True
    detect_anomaly: bool = False
    barebones: bool = False
    sync_batchnorm: bool = True
    reload_dataloaders_every_n_epochs: int = 0
    default_root_dir: Optional[str] = "./data/pytorch-lightning"

    def loggable_params(self) -> list[str]:
        return ["max_epochs", "max_steps", "strategy", "precision"]
    
    
@dataclass
class GPUDevConfig(TrainerConfig):
    max_epochs: int = 3
    accelerator: str = "cuda"  # auto, cpu, cuda, mps, tpu
    log_every_n_steps: int = 1
    limit_train_batches: float = 0.01
    limit_val_batches: float = 0.01
    limit_test_batches: float = 0.01
    logger: Optional[list[logger_schemas.LoggerConfig]] = field(default_factory=lambda: [logger_schemas.MLFlowLoggerConfig()])
    callbacks: Optional[list[callbacks_schemas.CallbackConfig]] = field(default_factory=lambda: [
        callbacks_schemas.ValidationF1ScoreBestModelCheckpointConfig(),
        callbacks_schemas.LastModelCheckpointConfig(),
        callbacks_schemas.LearningRateMonitorConfig()
    ])
    

@dataclass
class GPUProdConfig(TrainerConfig):
    max_epochs: int = 10
    accelerator: str = "cuda"  # auto, cpu, cuda, mps, tpu
    log_every_n_steps: int = 20
    logger: Optional[list[logger_schemas.LoggerConfig]] = field(default_factory=lambda: [logger_schemas.MLFlowLoggerConfig()])
    callbacks: Optional[list[callbacks_schemas.CallbackConfig]] = field(default_factory=lambda: [
        callbacks_schemas.ValidationF1ScoreBestModelCheckpointConfig(),
        callbacks_schemas.LastModelCheckpointConfig(),
        callbacks_schemas.LearningRateMonitorConfig()
    ])
    

def register_config() -> None:
    logger_schemas.register_config()
    callbacks_schemas.register_config()
    
    cs = ConfigStore.instance()
    cs.store(
        name="trainer_schema",
        group="tasks/trainer",
        node=TrainerConfig
    )