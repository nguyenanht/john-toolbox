# set default shell
SHELL := $(shell which bash)
FOLDER=$$(pwd)
SSL_DIR = ssl
# default shell options
.SHELLFLAGS = -c
.SILENT: ;
default: help;   # default target

# TODO : handle image_name for CPU
IMAGE_NAME=john-toolbox:latest
# cuda image
IMAGE_NAME_GPU=nguyenanht/cuda-python-poetry

IMAGE_RELEASER_NAME=release-changelog:latest
DOCKER_NAME = johntoolbox
DOCKER_NAME_GPU = johntoolboxgpu

DK = docker
DKC = docker-compose
DKC_CFG = -f docker-compose.yml
DKC_RUN = $(DKC) $(DKC_CFG) run --rm -T john_dev
SRC_DIR=john_toolbox

help: ## Display help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY: help


build: ## Build image
	$(DKC) $(DKC_CFG) build
.PHONY: build

install: ## First time: Build image, and install all the dependencies, including jupyter
	echo "generate env file"
	make env
	make stop
	echo "build docker images"
	make build
	echo "Installing dependencies"
	make deps
	echo "Configuring-pre-commit"
	make configure-pre-commit
	echo "Activating notebook extension"
	make up-notebook-extension
	echo "Changing current folder rights"
	sudo chmod -R 777 .cache
	make start
	make ps
.PHONY: install

# keep history of bash history
create-history:
	touch docker_history/.container_bash_history
.PHONY: create-history

start: ## Launch containers john_toolbox
	make create-history
	$(DKC) $(DKC_CFG) up -d --remove-orphans
.PHONY: start

stop: ## Stop and delete containers but leave network and volumes
	$(DKC) $(DKC_CFG) rm -f -v -s
	echo "stop and deleted containers but leave network and volumes."
.PHONY: stop

logs: ## Get log from a service
	@:$(call check_defined, svc, service)
	$(DKC) $(DKC_CFG) logs -t -f ${svc}
.PHONY: logs

destroy: ## destroy all containers, networks, volumes and orphaned containers
	read -p "Do you really want to destroy everything ? (y/n) : " resp ; \
	echo "" ; \
	if [ "$$resp" = "y" ]; then \
		$(DKC) $(DKC_CFG) down -v --remove-orphans --rmi all; \
	fi
.PHONY: destroy

deps: ## install dependencies
	$(DKC_RUN) poetry install --sync
.PHONY: deps

notebook:
	./open_nb.sh
.PHONY: notebook

#lab: ## Start the Jupyter lab (must be run from inside the container)
#	poetry run jupyter lab --allow-root --ip 0.0.0.0 --port ${PORT} --no-browser --notebook-dir .
#.PHONY: lab

ps: ## see docker running
	make ascii-logo
	docker ps
.PHONY: ps

ascii-logo:
	cat ascii_logo.txt
.PHONY: ascii-logo

services: ## List all possible services
	$(DKC) $(DKC_CFG) config --services
.PHONY: services

tests: ## To run tests inside the container (Go inside dev container to execute it)
	if [ -f /.dockerenv ]; then \
    	echo "Running inside docker"; \
		poetry run coverage run -m pytest -p no:cacheprovider tests/ ; \
	else \
		echo "You cannot run tests outside container."; \
	fi
.PHONY: tests

coverage: tests  ## To see tests coverage (Go inside dev container to execute it)
	if [ -f /.dockerenv ]; then \
    	echo "Running inside docker"; \
		poetry run coverage report; \
		poetry run coverage html; \
	else \
		echo "You cannot run coverage outside container."; \
	fi
	
.PHONY: coverage

docs: build ## Build and generate docs
	$(DKC_RUN) poetry run sphinx-build ./docs-scripts/source ./docs -b html
	$(DKC_RUN) touch ./docs/.nojekyll
.PHONY: doc

docs-prod: install ## Build and generate docs in production automatically
	$(DKC_RUN) poetry run sphinx-build ./docs-scripts/source ./docs -b html
	$(DKC_RUN) touch ./docs/.nojekyll
.PHONY: docprod

ci-pytest: install ## ci tests
	make env
	$(DKC_RUN) make tests
.PHONY: ci-pytest

prepare-release: build build-releaser ## Prepare release branch with changelog for given version
	echo "Executing script prepare-release.sh"
	./release-script/prepare-release.sh
.PHONY: prepare-release

do-release: build build-releaser ## Execute release branch with changelog for given version
	./release-script/do-release.sh
.PHONY: do-release

build-releaser: ## Build docker image for releaser
	echo "Building Dockerfile"
	docker build -f ./release-script/Dockerfile_changelog -t ${IMAGE_RELEASER_NAME} .
.PHONY: build-release

chown: ## Give rights to src and notebook
	sudo chown -R $$(whoami) notebooks john_toolbox tests data .git
	echo "right added to notebooks john_toolbox tests data directories"

env: ## Create .env file
	if [ ! -f .env  ]; then  cp .env.dist .env ; fi
	echo .env
.PHONY: env

up-notebook-extension: ## Activate useful notebook extensions
	$(DKC_RUN) poetry run jupyter contrib nbextension install --sys-prefix --symlink
	$(DKC_RUN) bash -c "poetry run jupyter nbextension enable autosavetime/main --sys-prefix \
	&&  poetry run jupyter nbextension enable tree-filter/index --sys-prefix \
	&&  poetry run jupyter nbextension enable splitcell/splitcell --sys-prefix \
	&&  poetry run jupyter nbextension enable toc2/main --sys-prefix \
	&&  poetry run jupyter nbextension enable toggle_all_line_numbers/main --sys-prefix \
	&&  poetry run jupyter nbextension enable cell_filter/cell_filter --sys-prefix \
	&&  poetry run jupyter nbextension enable code_prettify/autopep8 --sys-prefix"
.PHONY: up-notebook-extension

light-mode-theme: ## Activate light mode theme
	$(DKC_RUN) "poetry run jt -t grade3 -fs 95 -tfs 11 -nfs 115 -cellw 88% -T"
.PHONY: light-mode-theme

dark-mode-theme: ## Activate dark mode theme
	$(DKC_RUN) "poetry run jt -t monokai -fs 95 -tfs 11 -nfs 115 -cellw 88% -T"
.PHONY: dark-mode-theme

reset-theme: ## Activate dark mode theme
	$(DKC_RUN) "poetry run jt -r"
.PHONY: reset-theme

configure-pre-commit:
	# temporary fix for ubuntu20.04 Dockerfile_gpu
	$(DKC_RUN) bash -c "git config --global --add safe.directory /work && poetry run pre-commit install -f"
	make chown
	echo "Copy pre-commit configuration to .git/hooks"
	cp -a pre-commit/* .git/hooks/
.PHONY: configure-pre-commit

lint: ## flake8
	echo "executing command inside docker..."
	$(DKC_RUN) poetry run flake8 $(SRC_DIR)
	echo "apply flake8 done."
.PHONY: lint

black: ## black
	echo "executing command inside docker..."
	$(DKC_RUN) poetry run black $(SRC_DIR)
	echo "apply black done."
.PHONY: black

ssl: ## Create local cert
	if [ ! -f $(SSL_DIR)/cert-local-johntoolbox.key ]; then \
		sudo chmod -R 777 ./$(SSL_DIR); \
		docker run -it --rm -v ${PWD}/$(SSL_DIR):/work -w /work ubuntu bash -c "./create.sh"; \
		cd $(SSL_DIR); \
		./install-prerequisites.sh; \
		./add-to-chromium-firefox.sh; \
		./add-to-keychain.sh; \
		cd ..; \
		sudo chmod -R 777 ./$(SSL_DIR); \
	fi;
.PHONY: ssl

rm-ssl: ## remove local cert
	sudo rm -rf ssl/demoCA
	sudo rm -rf ssl/*.crt
	sudo rm -rf ssl/*.csr
	sudo rm -rf ssl/*.key
	sudo rm -rf ssl/*.pem
.PHONY: rm-ssl

push-docker-image-gpu:
	docker login
	@read -p "Enter tag name for GPU image ${IMAGE_NAME_GPU}:{tag}:" tag; \
	echo "building ${IMAGE_NAME_GPU}:$$tag"; \
	docker build -t  ${IMAGE_NAME_GPU}:$$tag . -f Dockerfile_gpu; \
	docker push  ${IMAGE_NAME_GPU}:$$tag;
.PHONY: push-docker-image-gpu
