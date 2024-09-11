from abc import abstractmethod
from typing import Any, Callable, Iterable, Optional, Union
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim import Optimizer

from jeffrey.data_modules.transformations import Transformation
from jeffrey.training.loss_functions import LossFunction
from jeffrey.training.schedulers import LightningScheduler
from jeffrey.models.models import Model
from jeffrey.utils.utils import get_logger


PartialOptimizerType = Callable[[Union[Iterable[Tensor], dict[str, Iterable[Tensor]]]], Optimizer]


class TrainingLightningModule(LightningModule):
    def __init__(
        self,
        model: Model,
        loss: LossFunction,
        optimizer: PartialOptimizerType,
        scheduler: Optional[LightningScheduler]
    ) -> None:
        super().__init__()
        
        self.model = model
        self.loss = loss
        self.partial_optimizer = optimizer
        self.scheduler = scheduler
        
        self.logging_logger = get_logger(self.__class__.__name__)
    
    # https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/core/module.html#LightningModule.configure_optimizers    
    def configure_optimizers(self) -> Union[Optimizer, tuple[list[Optimizer], list[dict[str, Any]]]]:
        optimizer = self.partial_optimizer(self.parameters())
        
        if self.scheduler is not None:
            scheduler = self.scheduler.configure_scheduler(
                optimizer=optimizer,
                estimated_stepping_batches=self.trainer.estimated_stepping_batches
            )
            return [optimizer], [scheduler]
        
        return optimizer
        
    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        ...
        
    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        ...
        
    @abstractmethod
    def get_transformation(self) -> Transformation:
        ...