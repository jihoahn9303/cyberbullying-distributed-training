from abc import abstractmethod
from typing import Any, Protocol

from torch import nn, Tensor
from lightning.pytorch import LightningModule

from jeffrey.models.models import Model
from jeffrey.models.transformations import Transformation


class EvaluationLightningModule(LightningModule):
    def __init__(self, model: Model) -> None:
        super().__init__()
        
        self.model = model
        
    @abstractmethod
    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        ...
        
    @abstractmethod
    def get_transformation(self) -> Transformation:
        ...
        

class PartialEvaluationLightningModuleType(Protocol):
    def __call__(self, model: nn.Module) -> EvaluationLightningModule:
        ...