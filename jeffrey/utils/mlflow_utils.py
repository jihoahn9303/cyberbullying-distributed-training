import dataclasses
import os
from contextlib import contextmanager
from typing import Any, Generator, Iterable, Optional, TYPE_CHECKING

import mlflow
from mlflow.pyfunc import PythonModel

from jeffrey.config_schemas.infrastructure.infrastructure_schema import MLFlowConfig
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
    
    
def get_client() -> mlflow.tracking.MlflowClient:
    return mlflow.tracking.MlflowClient(MLFLOW_TRACKING_URI)


def get_all_experiment_ids() -> list[str]:
    return [exp.experiment_id for exp in mlflow.search_experiments()]


def get_best_run() -> dict[str, Any]:
    best_runs = mlflow.search_runs(experiment_ids=get_all_experiment_ids(), filter_string="tag.best_run LIKE 'v%'")
    if len(best_runs) == 0:
        return {}

    indices = best_runs["tags.best_run"].str.split("v").str[-1].astype(int).sort_values()
    best_runs = best_runs.reindex(index=indices.index)
    best_runs_dict: dict[str, Any] = best_runs.iloc[-1].to_dict()
    return best_runs_dict


def log_model(
    mlflow_config: MLFlowConfig,
    new_best_run_tag: str,
    registered_model_name: str
) -> None:
    experiment_name = mlflow_config.experiment_name
    run_id = mlflow_config.run_id
    run_name = mlflow_config.run_name
    
    with activate_mlflow(
        experiment_name=experiment_name,
        run_id=run_id,
        run_name=run_name
    ) as _:
        mlflow.pyfunc.log_model(
            artifact_path="",
            python_model=DummyWrapper(),
            registered_model_name=registered_model_name
        )
        mlflow.set_tag(key="best_run", value=new_best_run_tag)
        
        
class DummyWrapper(PythonModel):
    def load_context(self, some_path: str) -> None:
        pass
    
    def predict(self, some_input: Any, some_other_parameter: Any) -> Optional[float]:
        pass