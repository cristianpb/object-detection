# include .env file
include .env

# this is usefull with most python apps in dev mode because if stdout is
# buffered logs do not shows in realtime
PYTHONUNBUFFERED=1
export

# compose command to merge production file and and dev/tools overrides
COMPOSE?=docker-compose -f docker-compose.yml

.env:
	cp .env.sample .env

config.yml:
	cp config.yml.sample config.yml

venv:
	@echo "Installing dependencies for $(CAMERA)"
	@if [ "${CAMERA}" = 'pi' ]; then \
		sudo apt install python3-dotenv python3-pandas python3-picamera python3-flask python3-redis python3-pip; \
		sudo pip3 install -e .; \
		mkdir venv; \
	elif [ "${CAMERA}" = 'jetson' ]; then \
		sudo apt install python3-dotenv python3-pandas python3-flask python3-redis python3-pip; \
		sudo pip3 install Cython; \
		sudo apt-get install protobuf-compiler libprotobuf-dev protobuf-compiler; \
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

dev: .env config.yml dist build
	@echo "Debug mode $(CAMERA) $(PORT)"
	@if [ "${CAMERA}" = 'pi' ]; then \
		DEBUG=1 FLASK_APP=backend/app.py flask run; \
	elif [ "${CAMERA}" = 'jetson' ]; then \
		DEBUG=1 FLASK_APP=backend/app.py flask run; \
	else \
		DEBUG=1 FLASK_APP=backend/app.py venv/bin/flask run; \
	fi

up: .env config.yml dist build
	@echo "Up mode $(CAMERA) $(PORT)"
	@if [ "${CAMERA}" = 'pi' ]; then \
		DEBUG="" FLASK_APP=backend/app.py flask run; \
	elif [ "${CAMERA}" = 'jetson' ]; then \
		DEBUG="" FLASK_APP=backend/app.py flask run; \
	else \
		DEBUG="" FLASK_APP=backend/app.py venv/bin/flask run; \
	fi

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
