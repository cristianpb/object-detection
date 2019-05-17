FROM python:3.7

# we create a user for our app
ARG uid=1001
RUN useradd factory --uid ${uid} --shell /bin/bash --home /home/factory
WORKDIR /home/factory

# extends PATH to use our venv first
ENV PATH=/home/factory/venv/bin/:$PATH

# copy our project
COPY ./app /home/factory

# use our user
USER factory

# run our application
CMD [ "python3", "-m", "flask_app.app" ]
