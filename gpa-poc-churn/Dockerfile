FROM python:3.7-slim-buster AS base_image

ARG APP_DIR=/usr/app/

USER root

RUN mkdir ${APP_DIR}

WORKDIR ${APP_DIR}

RUN apt-get update

RUN apt-get install -y build-essential

RUN pip install --upgrade pip

COPY requirements.txt ${APP_DIR}

# NOT WORKING
# This is needed for plotly to shoup on jupyterlab
# If you dont want this functionality, comment this section
#RUN apt -y install nodejs=12.0.0
#RUN apt -y install snapd
#RUN snap install node --channel=12/stable --classic
#RUN pip3 install jupyterlab "ipywidgets>=7.5"
#RUN jupyter labextension install jupyterlab-plotly@4.14.3

#RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.3
RUN pip3 install -r requirements.txt

###########################################################
# gcloud sdk
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
    apt-get update -y && \
    apt-get install google-cloud-sdk -y
###########################################################

COPY ./src ./src

RUN chmod -R 755 /usr/app