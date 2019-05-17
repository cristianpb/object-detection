# The Makefile defines all builds/tests steps

# current user id is usefull to be able to write in volumes
export UID:=$(shell id -u)

# include .env file
include .env

# compose command to merge production file and and dev/tools overrides
COMPOSE?=docker-compose -f docker-compose.yml

# docker run command used to build dependencies before images build
DOCKER_PIP?=docker run --rm -u $(UID) -e "HOME=/home/factory" -w "/home/factory" \
	-v "$(PWD)/app/:/home/factory" -v "$(HOME)/.cache:/.cache" python:3.7

export COMPOSE DOCKER_PIP 

# this is usefull with most python apps in dev mode because if stdout is
# buffered logs do not shows in realtime
PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED

pull:
	# pull latest images required for our build
	docker pull python:3.7

$(HOME)/.cache:
	mkdir -p $(HOME)/.cache

venv: $(HOME)/.cache
	# install some software with a dev image here
	$(DOCKER_PIP) python3 -m venv venv
	$(DOCKER_PIP) venv/bin/pip install -U -r requirements.txt

lock:
	# you may want something usefull to freeze your dependencies
	$(DOCKER_PIP) venv/bin/pip freeze > requirements-lock.txt

build: venv
	# build the docker images with the commit as tag
	docker build -t $(CI_REGISTRY_IMAGE)/python:$(CI_COMMIT_SHA) \
		-f docker/python/Dockerfile --build-arg=uid=$(UID) .

up:
	# run compose in background
	$(COMPOSE) up -d

dev:
	# run compose in background
	$(COMPOSE) up

ps:
	$(COMPOSE) ps

down:
	# stop compose
	$(COMPOSE) down --remove-orphans

test:

clean:
	# clean local folders
	rm -Rf venv
	# remove docker stuff
	$(COMPOSE) rm --stop --force
	# clean docker volumes
	docker volume prune -f || true
	# clean docker networks
	docker network prune -f || true
