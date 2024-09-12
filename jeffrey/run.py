import torch
import torch.distributed
from lightning import seed_everything
from hydra.utils import instantiate

from jeffrey.config_schemas.config_schema import Config
from jeffrey.utils.config_utils import get_config
from jeffrey.utils.torch_utils import get_local_rank
from jeffrey.utils.utils import get_logger


@get_config(config_path="../configs/automatically_generated", config_name="config", to_object=False, return_dict_config=True)
def run_tasks(config: Config) -> None:
    logger = get_logger(name=__file__)
    assert config.infrastructure.mlflow.run_id is not None, "Run id has to be set for training task"
    
    backend = "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{get_local_rank()}")
        backend = "nccl"

    torch.distributed.init_process_group(backend=backend)
    
    seed_everything(seed=config.seed, workers=True)
    
    for task_name, task_config in config.tasks.items():
        logger.info(f"Running task: {task_name}")
        task = instantiate(task_config)
        task.run(config=config, task_config=task_config)
        

if __name__ == "__main__":
    run_tasks()  # type: ignore
