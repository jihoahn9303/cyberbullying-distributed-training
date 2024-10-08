from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING, SI
from hydra.core.config_store import ConfigStore

from jeffrey.config_schemas.models import transformations_schemas
from jeffrey.utils.mixins import LoggerbleParamsMixin


@dataclass
class DataModuleConfig(LoggerbleParamsMixin):
    _target_: str = MISSING
    batch_size: int = MISSING
    shuffle: bool = False
    num_workers: int = 8
    pin_memory: bool = True
    drop_last: bool = True
    persistent_workers: bool = False
    
    def loggable_params(self) -> list[str]:
        return ["_target_", "batch_size"]
    

@dataclass
class TextClassificationDataModuleConfig(DataModuleConfig):
    _target_: str = "jeffrey.data_modules.data_modules.TextClassificationDataModule"
    train_df_path: str = MISSING
    valid_df_path: str = MISSING
    test_df_path: str = MISSING
    text_column_name: str = "cleaned_text"
    label_column_name: str = "label"
    transformation: transformations_schemas.TransformationConfig = MISSING
    

@dataclass
class ScrappedDataTextClassificationDataModuleConfig(TextClassificationDataModuleConfig):
    batch_size: int = 64
    train_df_path: str = "gs://jeffrey-data-versioning/data/processed/rebalanced_splits/train.parquet"
    valid_df_path: str = "gs://jeffrey-data-versioning/data/processed/rebalanced_splits/valid.parquet"
    test_df_path: str = "gs://jeffrey-data-versioning/data/processed/rebalanced_splits/test.parquet"
    transformation: transformations_schemas.TransformationConfig = SI("${..lightning_module.model.backbone.transformation}")

    
def register_config() -> None:
    transformations_schemas.register_config()
    
    cs = ConfigStore.instance()
    cs.store(
        name="text_classification_data_module_schema", 
        group="tasks/data_module",
        node=TextClassificationDataModuleConfig
    )

