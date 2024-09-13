import dataclasses
import os
from contextlib import contextmanager
from typing import Any, Generator, Iterable, Optional, TYPE_CHECKING

import mlflow

from jeffrey.utils.mixins import LoggerbleParamsMixin

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_ARTIFACT_STORE = os.getenv("MLFLOW_ARTIFACT_STORE")

if TYPE_CHECKING:
    from jeffrey.config_schemas.config_schema import Config


@contextmanager
def activate_mlflow(
    experiment_name: Optional[str] = None,
    run_id: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Iterable[mlflow.ActiveRun]:
    
    set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name, run_id=run_id) as run:
        yield run
    

def set_experiment(experiment_name: Optional[str] = None) -> None:
    if experiment_name is None:
        experiment_name = "Default"
    
    try:
        mlflow.create_experiment(name=experiment_name, artifact_location=MLFLOW_ARTIFACT_STORE)
    except mlflow.exceptions.RestException:
        pass
    finally:
        mlflow.set_experiment(experiment_name=experiment_name)
        
def log_artifacts_for_reproducibility() -> None:
    locations_to_store = [
        "./jeffrey",
        "./docker",
        "./pyproject.toml",
        "./poetry.lock"
    ]
    
    for location_to_store in locations_to_store:
        mlflow.log_artifact(
            local_path=location_to_store,
            artifact_path="reproduction"
        )
        
def log_training_hparams(config: "Config") -> None:
    logged_nodes = set()

    def loggable_params(node: Any, path: list[str]) -> Generator[tuple[str, Any], None, None]:
        # log target parameters in 'params' variable that will be sent to mlflow server
        if isinstance(node, LoggerbleParamsMixin) and id(node) not in logged_nodes:
            for param_name in node.loggable_params():
                yield ".".join(path + [param_name]), getattr(node, param_name)
            logged_nodes.add(id(node))
            
        # Search 'LoggerbleParamsMixin' based dataclass in children node
        children = None
        if isinstance(node, dict):
            children = node.items()
        if dataclasses.is_dataclass(node):
            children = ((f.name, getattr(node, f.name)) for f in dataclasses.fields(node))  # type: ignore

        # Base case: There is no children node
        if children is None:
            return
        
        # Call loggable_params() method recursively
        for key, val in children:
            for item in loggable_params(val, path + [key]):
                yield item

    params = dict(loggable_params(config, []))
    mlflow.log_params(params)