FROM python:3.8-slim AS base-image

ARG APP_DIR=/usr/app
USER root

# creates workdir folder
RUN mkdir ${APP_DIR}

# workdir setup
WORKDIR ${APP_DIR}
RUN apt-get update
RUN apt-get install -y build-essential

# copies requirements.txt to environment
COPY requirements.txt ${APP_DIR}

# installing python requirements - contains jupyter lab
RUN pip3 install -r requirements.txt

#pre-trained model package installation
RUN python -m spacy download pt

# copying content from base folder
COPY ./src ./src 