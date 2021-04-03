# set default shell
SHELL := $(shell which bash)
FOLDER=$$(pwd)
# default shell options
.SHELLFLAGS = -c
PORT=8884
.SILENT: ;
default: help;   # default target

IMAGE_NAME=john-toolbox:latest
DOCKER_NAME = johntoolbox
DOCKER_NAME_GPU = johntoolboxgpu
DOCKER_RUN = docker run  --rm  -v ${FOLDER}:/work -w /work --entrypoint bash -lc ${IMAGE_NAME} -c

build:
	echo "Building Dockerfile"
	docker build -t ${IMAGE_NAME} .
.PHONY: build

install: build ## First time: Build image, and install all the dependencies, including jupyter
	docker run --rm     -v ${FOLDER}:/work -w /work --entrypoint bash -lc ${IMAGE_NAME} -c 'poetry install'
	echo "Changing current folder rights"
	sudo chmod -R 777 .cache
.PHONY: install

build-gpu:
	echo "Building Dockerfile"
	docker build -t ${IMAGE_NAME} . -f Dockerfile_gpu
.PHONY: build-gpu

install-gpu: build-gpu ## First time: Build image gpu, and install all the dependencies, including jupyter
	docker run --gpus all --rm -v ${FOLDER}:/work -w /work --entrypoint bash -lc ${IMAGE_NAME} -c  'poetry install'
	echo "Changing current folder rights"
	sudo chmod -R 777 .cache
.PHONY: install-gpu

start: ## To get inside the container (can launch "poetry shell" from inside or "poetry add <package>")
	echo "Starting container ${IMAGE_NAME}"
	docker run --name $(DOCKER_NAME) --rm -it -v ${FOLDER}:/work -w /work -p ${PORT}:${PORT} -e "JUPYTER_PORT=${PORT}" ${IMAGE_NAME}
.PHONY: start

start-gpu: ## To get inside the gpu container (can launch "poetry shell" from inside or "poetry add <package>")
	echo "Starting container ${IMAGE_NAME}"
	docker run --name $(DOCKER_NAME_GPU) --gpus all --rm -it -v ${FOLDER}:/work -w /work -p ${PORT}:${PORT} -e "JUPYTER_PORT=${PORT}" ${IMAGE_NAME}
.PHONY: start-gpu

notebook: ## Start the Jupyter notebook (must be run from inside the container)
	poetry run jupyter contrib nbextension install
	poetry run jupyter nbextension enable autosavetime/main
	poetry run jupyter nbextension enable tree-filter/index
	poetry run jupyter nbextension enable splitcell/splitcell
	poetry run jupyter nbextension enable toc2/main
	poetry run jupyter nbextension enable toggle_all_line_numbers/main
	poetry run jupyter nbextension enable cell_filter/cell_filter
	poetry run jupyter nbextension enable code_prettify/autopep8
	poetry run jupyter notebook --allow-root --ip 0.0.0.0 --port ${PORT} --no-browser --notebook-dir .
	# &> /dev/null &
.PHONY: notebook

lab: ## Start the Jupyter notebook (must be run from inside the container)
	poetry run jupyter lab --allow-root --ip 0.0.0.0 --port ${PORT} --no-browser --notebook-dir .
.PHONY: notebook

help: ## Display help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY: help

ps: ## see docker running
	docker ps
.PHONY: ps

bash: ## Bash
	docker exec -it $(DOCKER_NAME) bash
.PHONY: bash

bash-gpu: ## Bash gpu
	docker exec -it $(DOCKER_NAME_GPU) bash
.PHONY: bash-gpu

tests: ## To run tests
	poetry run coverage run -m pytest -p no:cacheprovider tests/
.PHONY: tests

coverage: tests  ## To see tests coverage
	poetry run coverage report
	poetry run coverage html
.PHONY: coverage

docs: build ## Build and generate docs
	$(DOCKER_RUN) 'poetry run sphinx-build ./docs-scripts/source ./docs -b html'
	$(DOCKER_RUN) 'touch ./docs/.nojekyll'
.PHONY: doc

docs-prod: install ## Build and generate docs in production
	$(DOCKER_RUN) 'poetry run sphinx-build ./docs-scripts/source ./docs -b html'
	$(DOCKER_RUN) 'touch ./docs/.nojekyll'
.PHONY: docprod