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

celery:
	if [ "${CAMERA}" = 'pi' ]; then \
		celery -A backend.camera_pi worker -B --loglevel=INFO; \
	else \
		celery -A backend.camera_opencv worker -B --loglevel=INFO; \
	fi

up: .env
	echo "Using $(CAMERA) $(PORT)"
	DEBUG="" python3 backend/app.py
