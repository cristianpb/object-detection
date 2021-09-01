# include .env file
include .env

# this is usefull with most python apps in dev mode because if stdout is
# buffered logs do not shows in realtime
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
export

# compose command to merge production file and and dev/tools overrides
COMPOSE?=docker-compose -f docker-compose.yml

.env:
	cp .env.sample .env

config.yml:
	cp config.yml.sample config.yml

network:
	@echo "Create network"
	@docker network create isolated_nw 2> /dev/null; true

venv:
	@echo "Installing dependencies for $(PLATFORM)"
	@if [ "${PLATFORM}" = 'pi' ]; then \
		sudo apt install -y python3-dotenv python3-pandas python3-picamera python3-flask python3-redis python3-pip; \
		sudo pip3 install -e .; \
		mkdir venv; \
	elif [ "${PLATFORM}" = 'jetson' ]; then \
		sudo apt install -y python3-dotenv python3-redis python3-pip; \
		sudo pip3 install pandas flask; \
		sudo pip3 install Cython; \
		sudo apt-get install -y protobuf-compiler libprotobuf-dev protobuf-compiler; \
		export PATH="/usr/local/cuda/bin:$PATH"; \
		export CUDA_INC_DIR="/usr/local/cuda/include"; \
		pip3 install pycuda; \
		sudo pip3 install -e .; \
		mkdir venv; \
	else \
		python3 -m venv venv; \
		venv/bin/pip install -U -r requirements.txt; \
		venv/bin/pip install -e .; \
	fi

dist:
	git clone --single-branch --depth=1 --branch builds https://github.com/cristianpb/object-detection-frontend dist

frontend:
	git clone --branch master https://github.com/cristianpb/object-detection-frontend frontend

push:
	rsync -avz --exclude 'backend.egg-info' --exclude 'dist' --exclude '.env' --exclude 'git' --exclude 'imgs' --exclude 'models' --exclude '.mypy_cache' --exclude '.pytest_cache' --exclude 'venv' * jetson:~/object-detection/

models/ssd_mobilenet/frozen_inference_graph.pb:
	curl -o ssd_mobilenet.tar.gz http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
	tar xvzf ssd_mobilenet.tar.gz -C models/ssd_mobilenet --strip-components=1
	rm -rf ssd_mobilenet.tar.gz

models/yolo/yolov3-tiny.weights:
	curl -o models/yolo/yolov3-tiny.weights https://pjreddie.com/media/files/yolov3-tiny.weights

models/yolo/yolov3.weights:
	curl -o models/yolo/yolov3.weights https://pjreddie.com/media/files/yolov3.weights

build: venv models/ssd_mobilenet/frozen_inference_graph.pb

dev: .env config.yml dist build frontend
	@echo "Listening on port: $(NGINX_PORT)"
	@export EXEC_ENV=dev; $(COMPOSE) -f docker-compose-dev.yml up --build 

up: network .env config.yml dist build
	@echo "Listening on port: $(NGINX_PORT)"
	@export EXEC_ENV=prod; $(COMPOSE) up -d

down:
	@$(COMPOSE) -f docker-compose-dev.yml down

heroku: dist models/ssd_mobilenet/frozen_inference_graph.pb config.yml
	DEBUG="" FLASK_APP=backend/app.py flask run

nginx-dev:
	$(COMPOSE) -f docker-compose-dev.yml up -d nginx

nginx-up:
	$(COMPOSE) up -d nginx

nginx-down:
	$(COMPOSE) stop nginx

redis-up:
	$(COMPOSE) up -d redis

redis-down:
	$(COMPOSE) stop redis

docker-up: nginx-up redis-up

docker-down:
	$(COMPOSE) down

clean:
	rm -rf venv dist
