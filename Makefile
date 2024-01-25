# set default shell
SHELL := $(shell which bash)
GROUP_ID = $(shell id -g)
USER_ID = $(shell id -u)
FOLDER=$$(pwd)
SSL_DIR = ssl
# default shell options
.SHELLFLAGS = -c
MAKE_BIN=$(MAKE)
.SILENT: ;			   # no need for @
.ONESHELL: ;			 # recipes execute in same shell
.NOTPARALLEL: ;		  # wait for this target to finish
.EXPORT_ALL_VARIABLES: ; # send all vars to shell
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
	$(ENV) USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) $(DKC) $(DKC_CFG) build
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
	$(ENV) USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) $(DKC) $(DKC_CFG) up -d --remove-orphans
.PHONY: start

stop: ## Stop and delete containers but leave network and volumes
	$(ENV) USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) $(DKC) $(DKC_CFG) rm -f -v -s
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
	$(ENV) USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) $(DKC_RUN) poetry install --sync
.PHONY: deps

notebook:
	./open_nb.sh
.PHONY: notebook

lab: ## Start the Jupyter lab (must be run from inside the container)
	poetry run jupyter lab --allow-root --ip 0.0.0.0 --port ${PORT} --no-browser --notebook-dir .
.PHONY: lab

ps: ## see docker running
	make ascii-logo
	$(DKC) $(DKC_CFG) ps
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
	$(ENV) USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) $(DKC_RUN) poetry run sphinx-build ./docs-scripts/source ./docs -b html
	$(ENV) USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) $(DKC_RUN) touch ./docs/.nojekyll
.PHONY: doc

docs-prod: install ## Build and generate docs in production automatically
	$(DKC_RUN) poetry run sphinx-build ./docs-scripts/source ./docs -b html
	$(DKC_RUN) touch ./docs/.nojekyll
.PHONY: docprod

ci-pytest: install ## ci tests
	make env
	$(ENV) USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) $(DKC_RUN) make tests
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
	$(ENV) USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) $(DKC_RUN) bash -c "poetry run jupyter contrib nbextension install --sys-prefix --symlink"
	$(ENV) USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) $(DKC_RUN) bash -c "poetry run jupyter nbextension enable autosavetime/main --sys-prefix \
	&& poetry run jupyter nbextension enable --py widgetsnbextension \
	&&  poetry run jupyter nbextension enable tree-filter/index --sys-prefix \
	&&  poetry run jupyter nbextension enable splitcell/splitcell --sys-prefix \
	&&  poetry run jupyter nbextension enable toc2/main --sys-prefix \
	&&  poetry run jupyter nbextension enable toggle_all_line_numbers/main --sys-prefix \
	&&  poetry run jupyter nbextension enable cell_filter/cell_filter --sys-prefix \
	&&  poetry run jupyter nbextension enable code_prettify/autopep8 --sys-prefix \
	&&  poetry run jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip --sys-prefix \
	&&  poetry run jupyter nbextension enable jupyter-black-master/jupyter-black --sys-prefix"
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

configure-pre-commit: ## install precommit
	docker build --build-arg UID=$(USER_ID) --build-arg GID=$(GROUP_ID) -t pre-commit-image -f pre-commit/Dockerfile-pre-commit .

	if [ ! -d pre-commit/.cache ]; then
		mkdir pre-commit/.cache
	else
		echo "Directory pre-commit/.cache already exists, skipping creation."
	fi

	cp pre-commit/pre-commit .git/hooks/pre-commit
	echo "cp pre-commit/pre-commit .git/hooks/pre-commit"

	chmod +x .git/hooks/pre-commit
	echo "chmod +x .git/hooks/pre-commit"

	chmod +x pre-commit/script-pre-commit.sh
	echo "chmod +x pre-commit/script-pre-commit.sh"
.PHONY: configure-pre-commit


lint: ## 
	echo "executing command inside docker..."
	RUFF_EXIT_CODE=0
	$(ENV) USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) $(DKC_RUN) poetry run ruff check $(SRC_DIR) || RUFF_EXIT_CODE=$$?
	echo "apply ruff done."
	[ $$RUFF_EXIT_CODE -eq 0 ] || exit $$RUFF_EXIT_CODE
.PHONY: lint

format-all: ## format all
	echo "executing command inside docker..."
	$(ENV) USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) $(DKC_RUN) poetry run ruff check --fix $(SRC_DIR)
	echo "apply ruff done."
.PHONY: format-all

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
