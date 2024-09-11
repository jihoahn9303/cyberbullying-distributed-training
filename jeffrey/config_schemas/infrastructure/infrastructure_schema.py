from dataclasses import dataclass
from typing import Optional

from omegaconf import SI
from hydra.core.config_store import ConfigStore


@dataclass
class MLFlowConfig:
    mlflow_external_tracking_uri: str = SI("${oc.env:DEV_MLFLOW_TRACKING_URI,localhost:6101}")
    mlflow_internal_tracking_uri: str = SI("${oc.env:DEV_MLFLOW_INTERNAL_TRACKING_URI,localhost:6101}")
    experiment_name: Optional[str] = "Default"
    run_name: Optional[str] = None
    run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    experiment_url: str = SI(
        "${.mlflow_external_tracking_uri}/#/experiments/${.experiment_id}/runs/${.run_id}"
    )
    artifact_uri: Optional[str] = None
    block_mlflow: bool = False
    

@dataclass
class InfrastructureConfig:
    project_id: str = "e2eml-jiho-430901"
    zone: str = "asia-northeast3-c"
    # instance_group_creator: InstanceGroupCreatorConfig = InstanceGroupCreatorConfig()
    mlflow: MLFlowConfig = MLFlowConfig()
    
    
def register_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="infrastructure_schema",
        group="infrastructure",
        node=InfrastructureConfig
    )