# The Makefile defines all builds/tests steps

# include .env file
include .env

# this is usefull with most python apps in dev mode because if stdout is
# buffered logs do not shows in realtime
PYTHONUNBUFFERED=1
export

.env:
	cp .env.sample .env

venv:
	@echo "Installing dependencies for $(CAMERA)"
	@if [ "${CAMERA}" = 'pi' ]; then \
		sudo apt install python3-dotenv python3-pandas python3-picamera python3-flask python3-celery python3-redis; \
		mkdir venv; \
	elif [ "${CAMERA}" = 'jetson' ]; then \
		sudo apt install python3-dotenv python3-pandas python3-flask python3-celery python3-redis python3-pip; \
		sudo pip3 install Cython; \
		sudo apt-get install protobuf-compiler libprotobuf-dev protobuf-compiler; \
		pip3 install pycuda; \
		mkdir venv; \
	else \
		python3 -m venv venv; \
		venv/bin/pip install -U -r requirements.txt; \
		venv/bin/pip install -e .; \
	fi

dist:
	git clone --single-branch --branch builds https://github.com/cristianpb/object-detection-frontend dist

push:
	rsync -avz --exclude 'backend.egg-info' --exclude 'dist' --exclude '.env' --exclude 'git' --exclude 'imgs' --exclude 'models' --exclude '.mypy_cache' --exclude '.pytest_cache' --exclude 'venv' * jetson:~/object-detection/

models/ssd_mobilenet/frozen_inference_graph.pb:
	curl -o ssd_mobilenet.tar.gz http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
	tar xvzf ssd_mobilenet.tar.gz -C models/ssd_mobilenet --strip-components=1
	rm -rf ssd_mobilenet.tar.gz

models/yolo/yolov3-tiny.weights:
	#curl -o models/yolo/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
	curl -o models/yolo/yolov3-tiny.weights https://pjreddie.com/media/files/yolov3-tiny.weights

build: venv models/ssd_mobilenet/frozen_inference_graph.pb

dev: .env dist build
	@echo "Debug mode $(CAMERA) $(PORT)"
	@if [ "${CAMERA}" = 'pi' ]; then \
		DEBUG=1 python3 backend/app.py; \
	elif [ "${CAMERA}" = 'jetson' ]; then \
		DEBUG=1 python3 backend/app.py; \
	else \
		DEBUG=1 venv/bin/python3 backend/app.py; \
	fi

up: .env dist build
	@echo "Up mode $(CAMERA) $(PORT)"
	@if [ "${CAMERA}" = 'pi' ]; then \
		DEBUG="" python3 backend/app.py; \
	elif [ "${CAMERA}" = 'jetson' ]; then \
		DEBUG="" python3 backend/app.py; \
	else \
		DEBUG="" venv/bin/python3 backend/app.py; \
	fi

heroku: dist models/ssd_mobilenet/frozen_inference_graph.pb
	DEBUG="" python3 backend/app.py

celery:
	@echo "Launch Celery $(CAMERA)"
	@if [ "${CAMERA}" = 'pi' ]; then \
        python3 -m celery -A backend.camera_pi worker -B --loglevel=INFO; \
	elif [ "${CAMERA}" = 'jetson' ]; then \
        python3 -m celery -A backend.camera_jetson worker -P solo -c 1 --loglevel=INFO; \
	else \
		venv/bin/celery -A backend.camera_opencv worker -B --loglevel=INFO; \
	fi

celery_prod:
	@echo "Launch Celery as daemon $(CAMERA)"
	@if [ "${CAMERA}" = 'pi' ]; then \
        python3 -m celery -A backend.camera_pi worker -B --loglevel=ERROR --detach; \
	else \
		venv/bin/celery -A backend.camera_opencv worker -B --loglevel=ERROR --detach; \
	fi

flower:
	@echo "Launch Flower for $(CAMERA)"
	@if [ "${CAMERA}" = 'pi' ]; then \
		flower -A backend.camera_pi --address=0.0.0.0 --port=5555 --log-file-prefix=flower --url_prefix=flower; \
	else \
		venv/bin/flower -A backend.camera_opencv --address=0.0.0.0 --port=5555 --log-file-prefix=flower --logging=debug --url_prefix=flower; \
	fi

flower_prod:
	@echo "Launch Flower in background for $(CAMERA)"
	@if [ "${CAMERA}" = 'pi' ]; then \
		flower -A backend.camera_pi --address=0.0.0.0 --port=5555 --log-file-prefix=flower --url_prefix=flower & \
	else \
		venv/bin/flower -A backend.camera_opencv --address=0.0.0.0 --port=5555 --log-file-prefix=flower --logging=debug --url_prefix=flower & \
	fi

clean:
	rm -rf venv dist
