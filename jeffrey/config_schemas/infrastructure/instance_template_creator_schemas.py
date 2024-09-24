from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import SI

from jeffrey.infrastructure.instance_template_creator import VMType


# Selecting DL VM Image instance: https://cloud.google.com/deep-learning-vm/docs/images?hl=ko
@dataclass
class BootDiskConfig:
    project_id: str = "deeplearning-platform-release"
    image_name: str = "common-cu118-v20240730"
    size_gb: int = 50
    labels: Any = SI("${..labels}")


@dataclass
class VMConfig:
    machine_type: str = "g2-standard-8"
    accelerator_count: int = 1
    accelerator_type: str = "nvidia-l4"
    vm_type: VMType = VMType.STANDARD
    disks: list[str] = field(default_factory=lambda: [])


@dataclass
class VMMetadataConfig:
    zone: str = SI("${infrastructure.zone}")
    instance_group_name: str = SI("${infrastructure.instance_group_creator.name}")
    node_count: int = SI("${infrastructure.instance_group_creator.node_count}")
    disks: list[str] = SI("${..vm_config.disks}")
    docker_image: Optional[str] = SI("${docker_image}")
    mlflow_tracking_uri: str = SI("${infrastructure.mlflow.mlflow_internal_tracking_uri}")
    python_hash_seed: int = 42
    etcd_ip: Optional[str] = SI("${infrastructure.etcd_ip}")


@dataclass
class InstanceTemplateCreatorConfig:
    _target_: str = "jeffrey.infrastructure.instance_template_creator.InstanceTemplateCreator"
    # scope reference 1): https://developers.google.com/identity/protocols/oauth2/scopes?hl=ko
    # scope reference 2): https://cloud.google.com/dataproc/docs/concepts/configuring-clusters/service-accounts?hl=ko
    scopes: list[str] = field(
        default_factory=lambda: [
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/cloud.useraccounts.readonly",
            "https://www.googleapis.com/auth/cloudruntimeconfig",
        ]
    )
    network: str = SI("https://www.googleapis.com/compute/v1/projects/${.project_id}/global/networks/default")
    subnetwork: str = SI(
        "https://www.googleapis.com/compute/v1/projects/${.project_id}/regions/${infrastructure.region}/subnetworks/default"
    )
    startup_script_path: str = "scripts/vm_startup/task_runner_startup_script.sh"
    vm_config: VMConfig = VMConfig()
    boot_disk_config: BootDiskConfig = BootDiskConfig()
    vm_metadata_config: VMMetadataConfig = VMMetadataConfig()
    template_name: str = SI("${infrastructure.instance_group_creator.name}")
    project_id: str = SI("${infrastructure.project_id}")
    labels: dict[str, str] = field(default_factory=lambda: {"project": "e2eml-jiho-430901"})