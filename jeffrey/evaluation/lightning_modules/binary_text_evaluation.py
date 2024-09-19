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

from jeffrey.models.transformations import Transformation
from jeffrey.models.models import Model
from jeffrey.evaluation.lightning_modules.bases import EvaluationLightningModule
from jeffrey.utils.torch_utils import plot_confusion_matrix


class BinaryTextEvaluationLightningModule(EvaluationLightningModule):
    def __init__(
        self,
        model: Model
    ) -> None:
        super().__init__(model=model)
        
        self.test_accuracy = BinaryAccuracy()
        self.test_f1_score = BinaryF1Score()
        self.test_confusion_matrix = BinaryConfusionMatrix()

        self.test_step_outputs = defaultdict(list)
        
    def forward(self, texts: BatchEncoding) -> Tensor:
        return self.model(texts)
    
    def test_step(self, batch: Tuple[BatchEncoding, Tensor], batch_idx: int) -> None:
        texts, labels = batch
        logits = self(texts)  # (batch_size, 1)
        
        self.test_accuracy(logits, labels)
        self.test_f1_score(logits, labels)
        self.test_confusion_matrix(logits, labels)
        
        self.log(name="test_accuracy", value=self.test_accuracy, on_step=False, on_epoch=True)
        self.log(name="test_f1_score", value=self.test_f1_score, on_step=False, on_epoch=True)
        
        self.test_step_outputs["logits"].append(logits)
        self.test_step_outputs["labels"].append(labels)
    
    def on_test_epoch_end(self) -> None:
        all_logits = torch.stack(self.test_step_outputs["logits"])
        all_labels = torch.stack(self.test_step_outputs["labels"])
        
        confusion_matrix = self.test_confusion_matrix(all_logits, all_labels)
        figure = plot_confusion_matrix(confusion_matrix, class_names=["0", "1"])
        mlflow.log_figure(figure, artifact_file="test_confusion_matrix.png")
        
        self.test_step_outputs = defaultdict(list)
        
    def get_transformation(self) -> Transformation:
        return self.model.get_transformation()
