#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

if [[ "${IS_PROD_ENV}" == "true" ]]; then
    /usr/local/gcloud/google-cloud-sdk/bin/gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --tunnel-through-iap -- -N -L ${PROD_MLFLOW_SERVER_PORT}:localhost:${PROD_MLFLOW_SERVER_PORT}
else
    /docker/scripts/start-tracking-server.sh
fi
