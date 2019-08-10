# The Makefile defines all builds/tests steps

# include .env file
include .env

# this is usefull with most python apps in dev mode because if stdout is
# buffered logs do not shows in realtime
PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED

.env:
	cp .env.sample .env

build:
	pip3 install -r requirements.txt

dev: .env
	echo "Using $(CAMERA) $(PORT)"
	DEBUG=1 python3 backend/app.py

up: .env
	echo "Using $(CAMERA) $(PORT)"
	DEBUG="" python3 backend/app.py

celery:
	if [ "${CAMERA}" = 'pi' ]; then \
        python3 -m celery -A backend.camera_pi worker -B --loglevel=INFO -c 1; \
	else \
		python3 -m celery -A backend.camera_opencv worker -B --loglevel=INFO --detach; \
	fi

celery_prod:
	if [ "${CAMERA}" = 'pi' ]; then \
        python3 -m celery -A backend.camera_pi worker -B --loglevel=ERROR -c 1 --detach; \
	else \
		python3 -m celery -A backend.camera_opencv worker -B --loglevel=ERROR --detach; \
	fi
