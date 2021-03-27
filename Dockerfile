FROM python:3.7.9-slim-buster

RUN apt update \
    && apt install -y --no-install-recommends \
    curl \
    ca-certificates \
    bash-completion \
    libgomp1 \
    g++ \
    gcc \
    make \
    libopenblas-dev \
    python3-tk \
    && apt-get autoremove -y \
    && apt-get autoclean \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py -o /poetry.py \
    && python /poetry.py -y --version 0.12.17 && rm /poetry.py
ENV SHELL /bin/bash

ENV POETRY_CACHE /work/.cache/poetry
ENV PIP_CACHE_DIR /work/.cache/pip
ENV JUPYTER_RUNTIME_DIR /work/.cache/jupyter/runtime
ENV JUPYTER_CONFIG_DIR /work/.cache/jupyter/config

RUN $HOME/.poetry/bin/poetry config settings.virtualenvs.path $POETRY_CACHE

ENV PATH /root/.poetry/bin:/bin:/usr/local/bin:/usr/bin

CMD ["bash", "-l"]
# ENTRYPOINT ["poetry", "shell"]