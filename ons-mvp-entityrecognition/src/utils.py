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


def insert_new_metric(
    metrics_dict:dict,
    bucket:str='ons-ds-dev-entityrecognition',
    path:str='mpo/mvp/artifacts/model_info/metrics.parquet',
    model_id:str='model-default'
    ):
    
    metrics = pd.DataFrame(metrics_dict)
    metrics = pd.DataFrame(metrics.mean(axis=1), columns=['metrics']).transpose()
    metrics['variant'] = model_id
    obj_uri = f's3://{bucket}/{path}'
    
    try:
        s3_metrics = pd.read_parquet(obj_uri)
        s3_metrics = pd.concat([s3_metrics, metrics], axis=0)
    except:
        s3_metrics = metrics
    
    s3_metrics.drop_duplicates(keep='first', inplace=True)
    s3_metrics.sort_values(by=['variant'], inplace=True)
    s3_metrics.to_parquet(obj_uri)
    
    return s3_metrics


def load_metrics_from_s3(
    bucket:str='ons-ds-dev-entityrecognition',
    path:str='mpo/mvp/artifacts/model_info/metrics.parquet'
    ):
    
    obj_uri = f's3://{bucket}/{path}'
    metrics = pd.read_parquet(obj_uri)
    
    return metrics


def train_model(
    container,
    trainset_uri:str,
    validationset_uri:str,
    model_uri:str,
    hyperparameters:dict,
    job_name:str=None
    ):
    
    """
    Fun????o para treinamento do modelo preditivo. O modelo recebe dados em formato CSV.
    As vari??veis com sufixo '_uri' seguem a formata????o (s3://{bucket}/{file_path}).
    
    ---
    
    Vari??veis:
    - container (sagemaker obj): imagem contendo o modelo preditivo. \n
    - trainset_uri (str): caminho at?? o dataset de treino. \n
    - validationset_uri (str): caminho at?? o dataset de valida????o. \n
    - model_uri (str): caminho at?? o output do modelo (artefatos). \n
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
    Fun????o para declara????o de experimentos do sagemaker. 
    O nome do experimento ser?? automaticamente completado com um identificador cronol??gico (%Y-%m-%d-%H-%M-%S).
    
    ---
    
    Vari??veis:
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
    Fun????o para registro de experimentos do sagemaker. 
    O nome do experimento ser?? automaticamente incrementado com um identificador cronol??gico (%Y-%m-%d-%H-%M-%S) - nome do experimento = trial_prefix + identificador cronol??gico + trial_sufix.
    
    ---
    
    Vari??veis:
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
    Fun????o para excluir experimentos do sagemaker - limpar ambiente de experimentos.
    
    ---
    
    Vari??veis:
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
    print(f"Experimento {experiment_name} exclu??do com sucesso")
    
    return


def deploy_model(
    container,
    model_uri:str,
    endpoint_name:str,
    data_capture_config=None
    ):
    
    """
    Fun????o para implanta????o de um endpoint que ir?? hospedar a solu????o de ML.
    
    ---
    
    Vari??veis:
    - container (sagemaker obj): imagem contendo o modelo preditivo. \n
    - model_uri (str): caminho at?? o output do modelo (artefato model.tar.gz). \n
    - endpoint_name (str): label designado ao endpoint. \n
    - data_capture (bool): configura????o da pol??tica de armazenamento de dados enviados para o endpoint. 
    True => armazenar, False => n??o armazenar. Necess??rio para o monitoramento de modelos. Default = False. \n
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
    Fun????o para produzir configura????o de aquisi????o de dados de endpoints.
    
    ---
    
    Vari??veis:
    - storage_uri (str): caminho onde os dados capturados do endpoint ser??o armazenados. \n
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
    Fun????o para deletar um endpoint.
    
    ---
    
    Vari??veis:
    - endpoint_name (str): label designado ao endpoint. \n
    """
    
    sm.delete_endpoint(EndpointName=endpoint_name)
    
    print(f'Endpoint {endpoint_name} exclu??do com sucesso')
    
    return


def predict(
    input_data,
    endpoint_name:str,
    ):
    
    """
    Fun????o para predi????o com endpoints.
    
    ---
    
    Vari??veis:
    - input_data (pd.DataFrame): dados usados para a produ????o das infer??ncias. \n
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
    Fun????o para cria????o de um monitor de modelos e defini????o do baseline.
    
    ---
    
    Vari??veis:
    - bucket (str): nome do bucket no Simple Storage Service. \n
    - baseline_path (str): caminho at?? o diret??rio raiz do model monitor baseline. \n
    - trainset_uri (dict): caminho at?? o arquivo contendo os dados usados para treinamento do modelo. \n
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
    Fun????o para listar relat??rios dispon??veis.
    
    ---
    
    Vari??veis:
    - bucket (str): nome do bucket no Simple Storage Service. \n
    - baseline_results_path (str): caminho at?? o diret??rio de armazenamento de resultados de an??lise do baseline. \n
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
    Fun????o que recebe um monitor de dados do sagemaker e retorna o schema identificado.
    
    ---
    
    Vari??veis:
    - model_monitor (sagemaker.model_monitor): objeto de model monitor do sagemaker. \n
    """
    
    baseline_job = model_monitor.latest_baselining_job
    schema_df = pd.io.json.json_normalize(baseline_job.baseline_statistics().body_dict['features'])
    
    return schema_df


def get_model_monitor_constraints(
    model_monitor
    ):
    
    """
    Fun????o que recebe um monitor de dados do sagemaker e retorna o schema identificado.
    
    ---
    
    Vari??veis:
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
    Fun????o que cria um cronograma de monitoramento.
    
    ---
    
    Vari??veis:
    - model_monitor (sagemaker.model_monitor): objeto de model monitor do sagemaker. \n
    - endpoint (sagemaker.endpoint): objeto de endpoint do sagemaker. \n
    - monitor_schedule_name (str): nome do cronograma. \n
    - output_s3_uri (str): caminho de sa??da para as an??lises produzidas pelo cronograma. \n
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