########
#  dev #
########
FROM python:3.7-slim as dev

ARG APP_PATH
ARG APP_PORT

WORKDIR /${APP_PATH}

COPY LICENSE ./
COPY README.md ./
COPY requirements.txt ./
COPY setup.py ./
COPY config.yml ./

VOLUME /${APP_PATH}/backend
VOLUME /${APP_PATH}/tests
VOLUME /${APP_PATH}/dist
VOLUME /${APP_PATH}/models

RUN pip install --no-cache-dir -r requirements.txt

# Expose the listening port of your app
EXPOSE ${APP_PORT}

ENV FLASK_APP=backend/app.py
ENV DEBUG=1

CMD ["flask", "run", "--host=0.0.0.0"]

################################
# Step 2: "production" target #
################################
FROM dev as prod

ADD backend ./backend
ADD tests ./tests
ADD dist ./dist
ADD models ./models

COPY LICENSE ./
COPY README.md ./
COPY requirements.txt ./
COPY setup.py ./
COPY config.yml ./

ENV DEBUG=

CMD ["flask", "run", "--host=0.0.0.0"]
