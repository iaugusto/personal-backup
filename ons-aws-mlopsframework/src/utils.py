import os
import sys
import json
import time
import boto3
import numpy as np
import pandas as pd
from io import StringIO
import awswrangler as wr
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sagemaker
from sagemaker.model import Model
from sagemaker.session import Session
from sagemaker.analytics import ExperimentAnalytics
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor import CronExpressionGenerator
from sagemaker.model_monitor.dataset_format import DatasetFormat

from smexperiments.trial import Trial
from smexperiments.tracker import Tracker
from smexperiments.experiment import Experiment
from smexperiments.trial_component import TrialComponent

session = boto3.Session()
sm = session.client('sagemaker')
role = sagemaker.get_execution_role()
account_id = session.client('sts').get_caller_identity()['Account']


def train_model(
    container,
    trainset_uri:str,
    validationset_uri:str,
    model_uri:str,
    hyperparameters:dict,
    job_name:str=None
    ):
    
    """
    Função para treinamento do modelo preditivo. O modelo recebe dados em formato CSV.
    As variáveis com sufixo '_uri' seguem a formatação (s3://{bucket}/{file_path}).
    
    ---
    
    Variáveis:
    - container (sagemaker obj): imagem contendo o modelo preditivo. \n
    - trainset_uri (str): caminho até o dataset de treino. \n
    - validationset_uri (str): caminho até o dataset de validação. \n
    - model_uri (str): caminho até o output do modelo (artefatos). \n
    - hyperparameters (dict): hyperparametros do modelo. \n
    - job_name (str): opcional - nome do 'trial' para experimentos do sagemaker. \n
    """
    
    s3_input_train = sagemaker.inputs.TrainingInput(s3_data=trainset_uri, content_type='csv')
    s3_input_validation = sagemaker.inputs.TrainingInput(s3_data=validationset_uri, content_type='csv')
        
    model_ = sagemaker.estimator.Estimator(
        container,
        role=role,
        instance_count=2,
        instance_type='ml.m5.4xlarge',
        output_path=model_uri,
        enable_sagemaker_metrics=True
    )
    
    model_.set_hyperparameters(
        **hyperparameters
    )
    
    if job_name:
        
        model_.fit({
            'train':s3_input_train, 
            'validation':s3_input_validation,
        },
            job_name=job_name,
            experiment_config={
                'TrialName':job_name,
                'TrialComponentDisplayName':job_name
            },
            wait=True
        )

    else:
        
        model_.fit({
            'train':s3_input_train, 
            'validation':s3_input_validation,
        })
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Treinamento do modelo finalizado em {current_time}.')
    
    return model_


def create_experiment(
    experiment_prefix:str
    ):
    
    """
    Função para declaração de experimentos do sagemaker. 
    O nome do experimento será automaticamente completado com um identificador cronológico (%Y-%m-%d-%H-%M-%S).
    
    ---
    
    Variáveis:
    - experiment_prefix (str): nome do experimento. \n
    """
    
    experiment_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    experiment = Experiment.create(
        experiment_name=f"{experiment_prefix}-{experiment_time}",
        description=f"Framework Experiment - {experiment_prefix}",
        sagemaker_boto_client=sm
    )
    
    return experiment


def log_trial(
    experiment_name,
    trial_prefix:str,
    trial_sufix:str
    ):
    
    """
    Função para registro de experimentos do sagemaker. 
    O nome do experimento será automaticamente incrementado com um identificador cronológico (%Y-%m-%d-%H-%M-%S) - nome do experimento = trial_prefix + identificador cronológico + trial_sufix.
    
    ---
    
    Variáveis:
    - experiment_name (str) : nome do experimento do sagemaker. \n
    - trial_prefix (str): prefixo para o experimento a ser registrado. \n
    - trial_sufix (str): sufixo para o experimento a ser registrado. \n
    """
    
    trial_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    trial_name = f'{trial_prefix}-{trial_time}-{trial_sufix}'
    trial_ = Trial.create(
        trial_name=trial_name,
        experiment_name=experiment_name,
        sagemaker_boto_client=sm
    )
    
    return trial_, trial_name


def delete_experiment(
    experiment_name:str
    ):
    
    """
    Função para excluir experimentos do sagemaker - limpar ambiente de experimentos.
    
    ---
    
    Variáveis:
    - experiment (str) : nome do experimento do sagemaker. \n
    """
    
    experiment = Experiment.load(
        experiment_name=experiment_name
        )
    
    for trial_summary in experiment.list_trials():
        trial = Trial.load(trial_name=trial_summary.trial_name)
        for trial_component_summary in trial.list_trial_components():
            tc = TrialComponent.load(
                trial_component_name=trial_component_summary.trial_component_name)
            trial.remove_trial_component(tc)
            try:
                tc.delete()
            except:
                continue
            time.sleep(.5)
        trial.delete()
        experiment_name = experiment.experiment_name
    experiment.delete()
    print(f"Experimento {experiment_name} excluído com sucesso")
    
    return


def deploy_model(
    container,
    model_uri:str,
    endpoint_name:str,
    data_capture_config=None
    ):
    
    """
    Função para implantação de um endpoint que irá hospedar a solução de ML.
    
    ---
    
    Variáveis:
    - container (sagemaker obj): imagem contendo o modelo preditivo. \n
    - model_uri (str): caminho até o output do modelo (artefato model.tar.gz). \n
    - endpoint_name (str): label designado ao endpoint. \n
    - data_capture (bool): configuração da política de armazenamento de dados enviados para o endpoint. 
    True => armazenar, False => não armazenar. Necessário para o monitoramento de modelos. Default = False. \n
    """
    
    model_ = Model(
        image_uri=container, 
        model_data=model_uri, 
        role=role
        )
    
    if data_capture_config:
        model_.deploy(
            initial_instance_count=1,
            instance_type='ml.m4.xlarge',
            endpoint_name=endpoint_name,
            data_capture_config=data_capture_config
            )
    else:
        model_.deploy(
            initial_instance_count=1,
            instance_type='ml.m4.xlarge',
            endpoint_name=endpoint_name
            )
    
    print(f'Endpoint {endpoint_name} criado com sucesso')
    
    return


def get_data_capture_config(
    storage_uri:str,
    sampling_percentage:int=100
    ):
    
    """
    Função para produzir configuração de aquisição de dados de endpoints.
    
    ---
    
    Variáveis:
    - storage_uri (str): caminho onde os dados capturados do endpoint serão armazenados. \n
    - sampling_percentage (int): porcentagem dos dados fornecidos ao endpoint para armazenar. \n
    """
    
    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=sampling_percentage,
        destination_s3_uri=storage_uri
    )
    
    return data_capture_config


def delete_endpoint(
    endpoint_name:str
    ):
    
    """
    Função para deletar um endpoint.
    
    ---
    
    Variáveis:
    - endpoint_name (str): label designado ao endpoint. \n
    """
    
    sm.delete_endpoint(EndpointName=endpoint_name)
    
    print(f'Endpoint {endpoint_name} excluído com sucesso')
    
    return


def predict(
    input_data,
    endpoint_name:str,
    ):
    
    """
    Função para predição com endpoints.
    
    ---
    
    Variáveis:
    - input_data (pd.DataFrame): dados usados para a produção das inferências. \n
    - endpoint_name (str): label designado ao endpoint. \n
    """
    
    runtime = boto3.Session().client('sagemaker-runtime')
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name, 
        ContentType='text/csv', 
        Body=input_data.to_csv(header=False, index=False).encode("utf-8")
        )
    dict_response = response['Body'].read().decode('utf-8')
    
    output = input_data.copy()
    output['predicted'] = dict_response.split(',')
    output['predicted'] = output['predicted'].apply(lambda x:int(float(x)))
    
    return output


def create_model_monitor_baseline(
    bucket:str,
    baseline_path:str,
    trainset_uri:str
    ):
    
    """
    Função para criação de um monitor de modelos e definição do baseline.
    
    ---
    
    Variáveis:
    - bucket (str): nome do bucket no Simple Storage Service. \n
    - baseline_path (str): caminho até o diretório raiz do model monitor baseline. \n
    - trainset_uri (dict): caminho até o arquivo contendo os dados usados para treinamento do modelo. \n
    """

    baseline_data_uri = f's3://{bucket}/{baseline_path}/data'
    baseline_results_uri = f's3://{bucket}/{baseline_path}/results'
    print(f'baseline_data_uri: {baseline_data_uri}')
    print(f'baseline_results_uri: {baseline_results_uri}')
    
    model_monitor = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        volume_size_in_gb=20,
        max_runtime_in_seconds=1800
    )

    model_monitor.suggest_baseline(
        baseline_dataset=trainset_uri,
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=baseline_results_uri,
        wait=True
    )
    
    return model_monitor


def list_model_monitor_report_files(
    bucket:str,
    baseline_path:str
    ):
    
    """
    Função para listar relatórios disponíveis.
    
    ---
    
    Variáveis:
    - bucket (str): nome do bucket no Simple Storage Service. \n
    - baseline_results_path (str): caminho até o diretório de armazenamento de resultados de análise do baseline. \n
    """
    
    baseline_results_path = f'{baseline_path}/results'
    s3_client = session.client('s3')
    result = s3_client.list_objects(Bucket=bucket, Prefix=baseline_results_path)
    report_files = [report_file.get('Key') for report_file in result.get('Contents')]
    
    return report_files


def get_model_monitor_schema(
    model_monitor
    ):
    
    """
    Função que recebe um monitor de dados do sagemaker e retorna o schema identificado.
    
    ---
    
    Variáveis:
    - model_monitor (sagemaker.model_monitor): objeto de model monitor do sagemaker. \n
    """
    
    baseline_job = model_monitor.latest_baselining_job
    schema_df = pd.io.json.json_normalize(baseline_job.baseline_statistics().body_dict['features'])
    
    return schema_df


def get_model_monitor_constraints(
    model_monitor
    ):
    
    """
    Função que recebe um monitor de dados do sagemaker e retorna o schema identificado.
    
    ---
    
    Variáveis:
    - model_monitor (sagemaker.model_monitor): objeto de model monitor do sagemaker. \n
    """
    
    baseline_job = model_monitor.latest_baselining_job
    constraints_df = pd.io.json.json_normalize(baseline_job.suggested_constraints().body_dict['features'])
    
    return constraints_df


def create_monitoring_schedule(
    model_monitor,
    endpoint,
    monitor_schedule_name:str,
    output_s3_uri:str
    ):
    
    """
    Função que cria um cronograma de monitoramento.
    
    ---
    
    Variáveis:
    - model_monitor (sagemaker.model_monitor): objeto de model monitor do sagemaker. \n
    - endpoint (sagemaker.endpoint): objeto de endpoint do sagemaker. \n
    - monitor_schedule_name (str): nome do cronograma. \n
    - output_s3_uri (str): caminho de saída para as análises produzidas pelo cronograma. \n
    """

    model_monitor.create_monitoring_schedule(
        monitor_schedule_name=monitor_schedule_name,
        endpoint_input=endpoint,
        output_s3_uri=output_s3_uri,
        statistics=model_monitor.baseline_statistics(),
        constraints=model_monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True
        )
    
    return model_monitor