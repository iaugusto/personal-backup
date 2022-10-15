# ons-mvp-entity-recognition

## Description

This repository contains all the development of the Entity Recognition Engine (MPO-phase2). 

## Directory system

* artifacts: The artifacts folder serves as a reference directory, where the models will be saved during training.
* endpoint: The endpoint dir contains the predictor.py (flask api) and configuration files for building a custom sagemaker endpoint containing two models - the predefined entity recognition engine and the later proposed inequalities recognition engine.
* glue: The ons-ds-prod-entityrecognition-gluejob.py script is used to download the dependencies from S3, install the necessary python packages and run the training/retraining sequence. This script can be trigger by the arrival of new files to a specific directory or according to a schedule (the later seems to be the best option for this particular solution).
* lambda: The lambda_function.py script is used to link the API Gateway request to the sagemaker endpoint, that can only be accessed from within AWS (or externally using boto3).
* notebooks: This directory contains all the experimentation, development and testing notebooks used throughout the project.
* pipeline: This directory contains a preliminary implementation of a custom sagemaker pipeline for the automatic retraining sequence - later replaced with a GlueJob.
* src: This directory contains all the source code including the training sequence (train.py).

## Observations

The solution requires the correct setup of the AWS Infraestructure (IAM policies and roles, API Gateway, AWS Lambda and AWS Glue). 

The solution also requires the setup of the input files in the S3 bucket and the configuration of the retraining sequence according to the input bucket - the code is entirely customizable but follows, by default, a labeling pattern (according to the best governance practices), which allows its use in dev, hom and prod environments.

## To Do's

* If desirable, use the contents of the pipeline dir to implement a sagemaker pipeline.
* Implement a token access policy for api gateway and lambda integrations instead of a static authentication code.
    * This is not a solution that can leak sensitive data, so one has to weight in the pros and cons.