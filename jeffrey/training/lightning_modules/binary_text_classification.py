from typing import Dict, Optional, Tuple
from collections import defaultdict

import mlflow
from torch import Tensor
import torch
from transformers import BatchEncoding
from torchmetrics.classification import (
    BinaryAccuracy, 
    BinaryF1Score, 
    BinaryConfusionMatrix
)

from jeffrey.data_modules.transformations import Transformation
from jeffrey.models.models import Model
from jeffrey.training.lightning_modules.bases import PartialOptimizerType, TrainingLightningModule
from jeffrey.training.loss_functions import LossFunction
from jeffrey.training.schedulers import LightningScheduler
from jeffrey.utils.torch_utils import plot_confusion_matrix


class BinaryTextClassificationLightningModule(TrainingLightningModule):
    def __init__(
        self,
        model: Model,
        loss: LossFunction,
        optimizer: PartialOptimizerType,
        scheduler: Optional[LightningScheduler]
    ) -> None:
        super().__init__(model=model, loss=loss, optimizer=optimizer, scheduler=scheduler)
        
        self.training_accuracy = BinaryAccuracy()
        self.validation_accuracy = BinaryAccuracy()
        
        self.training_f1_score = BinaryF1Score()
        self.validation_f1_score = BinaryF1Score()
        
        self.training_confusion_matrix = BinaryConfusionMatrix()
        self.validation_confusion_matrix = BinaryConfusionMatrix()
        
        self.training_step_outputs = defaultdict(list)
        self.validation_step_outputs = defaultdict(list)
        
    def forward(self, texts: BatchEncoding) -> Tensor:
        return self.model(texts)
    
    def training_step(self, batch: Tuple[BatchEncoding, Tensor], batch_idx: int) -> Tensor:
        texts, labels = batch
        logits = self(texts)  # (batch_size, 1)
        
        loss = self.loss(logits, labels)
        self.log(name="loss", value=loss, sync_dist=True)
        
        self.training_accuracy(logits, labels)
        self.training_f1_score(logits, labels)
        self.training_confusion_matrix(logits, labels)
        
        self.log(name="training_accuracy", value=self.training_accuracy, on_step=False, on_epoch=True)
        self.log(name="training_f1_score", value=self.training_f1_score, on_step=False, on_epoch=True)
        
        self.training_step_outputs["logits"].append(logits)
        self.training_step_outputs["labels"].append(labels)
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        all_logits = torch.stack(self.training_step_outputs["logits"])
        all_labels = torch.stack(self.training_step_outputs["labels"])
        
        confusion_matrix = self.training_confusion_matrix(all_logits, all_labels)
        figure = plot_confusion_matrix(confusion_matrix, class_names=["0", "1"])
        mlflow.log_figure(figure, artifact_file="training_confusion_matrix.png")
        
        self.training_step_outputs = defaultdict(list)
    
    def validation_step(self, batch: Tuple[BatchEncoding, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        texts, labels = batch
        logits = self(texts)  # (batch_size, 1)
        
        loss = self.loss(logits, labels)
        self.log(name="validation_loss", value=loss, sync_dist=True)
        
        self.validation_f1_score(logits, labels)
        self.validation_accuracy(logits, labels)
        
        self.log(name="validation_accuracy", value=self.validation_accuracy, on_step=False, on_epoch=True)
        self.log(name="validation_f1_score", value=self.validation_f1_score, on_step=False, on_epoch=True)
        
        self.validation_step_outputs["logits"].append(logits)
        self.validation_step_outputs["labels"].append(labels)
        
        return {"loss": loss, "predictions": logits, "labels": labels}
    
    def on_validation_epoch_end(self) -> None:
        all_predictions = torch.stack(self.validation_step_outputs["logits"])
        all_labels = torch.stack(self.validation_step_outputs["labels"])
        
        confusion_matrix = self.validation_confusion_matrix(all_predictions, all_labels)
        figure = plot_confusion_matrix(confusion_matrix, class_names=["0", "1"])
        mlflow.log_figure(figure, artifact_file="validation_confusion_matrix.png")
        
        self.validation_step_outputs = defaultdict(list)
    
    def get_transformation(self) -> Transformation:
        return self.model.get_transformation()
