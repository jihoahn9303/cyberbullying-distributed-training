import os
from contextlib import contextmanager
from typing import Iterable, Optional

import mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_ARTIFACT_STORE = os.getenv("MLFLOW_ARTIFACT_STORE")


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
    
    # if mlflow.get_experiment_by_name(experiment_name) is not None:
    #     print(f"Experiment {experiment_name} is already exists.")
    #     pass
    # else:
    #     print(f"Creating new experiment {experiment_name}")
    #     mlflow.create_experiment(
    #         experiment_name,
    #         artifact_location=MLFLOW_ARTIFACT_STORE
    #     )

    # mlflow.set_experiment(experiment_name)