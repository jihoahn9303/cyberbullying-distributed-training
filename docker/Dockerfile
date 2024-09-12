FROM python:3.10-slim

ARG USER_ID
ARG USER_NAME

ENV HOME=/home/${USER_NAME} \
    VIRTUAL_ENV=/home/${USER_NAME}/venv 
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    PATH="/usr/local/gcloud/google-cloud-sdk/bin:${VIRTUAL_ENV}/bin:${HOME}/.local/bin:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH}" \
    BUILD_POETRY_LOCK="${HOME}/poetry.lock.build"


RUN apt-get -qq update \
    && apt-get -qq -y install vim gcc curl git build-essential libb64-dev software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq -y clean

RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-462.0.1-linux-x86_64.tar.gz \
    && mkdir -p /usr/local/gcloud \
    && tar -C /usr/local/gcloud -zxvf google-cloud-cli-462.0.1-linux-x86_64.tar.gz \
    && /usr/local/gcloud/google-cloud-sdk/install.sh --usage-reporting false --command-completion true --bash-completion true --path-update true --quiet

RUN addgroup --system --gid ${USER_ID} ${USER_NAME} \
    && useradd --system -m --no-log-init \
    --home-dir ${HOME} \
    --uid ${USER_ID} \
    --gid ${USER_NAME} \
    --groups ${USER_NAME} ${USER_NAME}

RUN chown -R ${USER_NAME}:${USER_NAME} ${HOME}
RUN mkdir -p /app /mlflow-cyberbullying-artifact-store \
    && chown -R ${USER_NAME}:${USER_NAME} /app /tmp /mlflow-cyberbullying-artifact-store
RUN curl -sSL https://install.python-poetry.org | python3 -

USER ${USER_NAME}
COPY pyproject.toml *.lock /app/
WORKDIR /app

RUN poetry config virtualenvs.create false \
    && python3.10 -m venv ${VIRTUAL_ENV} \
    && poetry install --only main --no-interaction \
    && poetry run pip install \
    torch==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html \
    && cp poetry.lock ${BUILD_POETRY_LOCK} \
    && rm -rf ${HOME}/.cache/*

USER root    
COPY --chown=${USER_NAME}:${USER_NAME} ./docker/scripts/* /docker/scripts/
RUN chmod a+x /docker/scripts/*.sh

USER ${USER_NAME}
COPY . /app/

CMD ["/docker/scripts/startup-script.sh"]
