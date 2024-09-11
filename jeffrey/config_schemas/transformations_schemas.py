from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class TransformationConfig:
    _target_: str = MISSING
    
    
@dataclass
class HuggingFaceTokenizationTransformationConfig(TransformationConfig):
    _target_: str = "jeffrey.data_modules.transformations.HuggingFaceTokenizationTransformation"
    pretrained_tokenizer_name_or_path: str = MISSING
    max_sequence_len: int = MISSING
    
    
@dataclass
class CustomHuggingFaceTokenizationTransformationConfig(HuggingFaceTokenizationTransformationConfig):
    pretrained_tokenizer_name_or_path: str = "gs://jeffrey-data-versioning/data/processed/rebalanced_splits/trained_tokenizer"
    max_sequence_len: int = 100
    

def register_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="huggingface_tokenization_transformation_schema",
        group="tasks/data_module/transformation",
        node=HuggingFaceTokenizationTransformationConfig
    )