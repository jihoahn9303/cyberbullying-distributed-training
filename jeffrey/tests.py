import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from jeffrey.config_schemas.training.training_task_schemas import register_config
from jeffrey.models.common.exporter import TarModelLoader

register_config()


# @hydra.main(config_name="test_training_task_schema", version_base=None)
# def main(config: DictConfig) -> None:
#     print(60 * '#')
#     print(OmegaConf.to_yaml(config))
#     print(60 * '#')
    
    # model = instantiate(config)
    
    # texts = ["Hello, how are you?"]
    # encodings = model.backbone.transformation(texts)
    
    # output = model(encodings)
    # print(f"{output.shape=}")
    
def main() -> None:
    model_loader = TarModelLoader("/mlflow-cyberbullying-artifact-store/0/a25c93c4fe7144d3ad59c2f9240936f5/artifacts/exported_model.tar.gz")
    model = model_loader.load()
    
    print(model)
    
    
if __name__ == "__main__":
    main()