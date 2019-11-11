# The Makefile defines all builds/tests steps

# include .env file
include .env

# this is usefull with most python apps in dev mode because if stdout is
# buffered logs do not shows in realtime
PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED

.env:
	cp .env.sample .env

venv:
	@echo "Installing dependencies for $(CAMERA)"
	@if [ "${CAMERA}" = 'pi' ]; then \
		sudo apt install python3-dotenv python3-pandas python3-pandas python3-picamera python3-flask python3-celery python3-redis; \
		sudo pip3 install -r requirements.txt; \
		mkdir venv; \
	else \
		python3 -m venv venv; \
		venv/bin/pip install -U -r requirements.txt; \
		venv/bin/pip install -e .; \
	fi

dist:
	git clone --single-branch --branch builds https://github.com/cristianpb/object-detection-frontend dist

models/ssd_mobilenet/frozen_inference_graph.pb:
	curl -o ssd_mobilenet.tar.gz http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
	tar xvzf ssd_mobilenet.tar.gz -C models/ssd_mobilenet --strip-components=1
	rm -rf ssd_mobilenet.tar.gz

build: venv models/ssd_mobilenet/frozen_inference_graph.pb

dev: .env dist build
	@echo "Debug mode $(CAMERA) $(PORT)"
	@if [ "${CAMERA}" = 'pi' ]; then \
		DEBUG=1 python3 backend/app.py; \
	else \
		DEBUG=1 venv/bin/python3 backend/app.py; \
	fi

up: .env dist build
	@echo "Up mode $(CAMERA) $(PORT)"
	@if [ "${CAMERA}" = 'pi' ]; then \
		DEBUG="" python3 backend/app.py; \
	else \
		DEBUG="" venv/bin/python3 backend/app.py; \
	fi

celery:
	@echo "Launch celery $(CAMERA)"
	@if [ "${CAMERA}" = 'pi' ]; then \
        python3 -m celery -A backend.camera_pi worker -B --loglevel=INFO; \
	else \
		venv/bin/celery -A backend.camera_opencv worker -B --loglevel=INFO; \
	fi

celery_prod:
	@echo "Launch celery as daemon $(CAMERA)"
	@if [ "${CAMERA}" = 'pi' ]; then \
        python3 -m celery -A backend.camera_pi worker -B --loglevel=ERROR --detach; \
	else \
		venv/bin/celery -A backend.camera_opencv worker -B --loglevel=ERROR --detach; \
	fi

clean:
	rm -rf venv dist
