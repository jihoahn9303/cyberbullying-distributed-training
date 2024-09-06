from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from jeffrey.config_schemas import transformations_schemas


@dataclass
class DataModuleConfig:
    _target_: str = MISSING
    batch_size: int = MISSING
    shuffle: bool = False
    num_workers: int = 8
    pin_memory: bool = True
    drop_last: bool = True
    persistent_workers: bool = False
    

@dataclass
class TextClassificationDataModuleConfig(DataModuleConfig):
    _target_: str = "jeffrey.data_modules.data_modules.TextClassificationDataModule"
    train_df_path: str = MISSING
    valid_df_path: str = MISSING
    test_df_path: str = MISSING
    text_column_name: str = "cleaned_text"
    label_column_name: str = "label"
    transformation: transformations_schemas.TransformationConfig = MISSING
    
    
def register_config() -> None:
    transformations_schemas.register_config()
    
    cs = ConfigStore.instance()
    cs.store(
        name="text_classification_data_module_schema", 
        group="tasks/data_module",
        node=TextClassificationDataModuleConfig
    )

