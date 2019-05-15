# The Makefile defines all builds/tests steps

# include .env file
include .env

# this is usefull with most python apps in dev mode because if stdout is
# buffered logs do not shows in realtime
PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED

build:
	pip3 install -r requirements.txt

dev:
	echo "Using $(CAMERA) $(PORT)"
	DEBUG=1 python3 app/app.py

up:
	echo "Using $(CAMERA) $(PORT)"
	DEBUG="" python3 app/app.py
