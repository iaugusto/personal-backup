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
from sagemaker.model_monitor.data_capture_config import DataCaptureConfig

import utils

session = boto3.Session()
sm = session.client('sagemaker')
role = sagemaker.get_execution_role()
account_id = session.client('sts').get_caller_identity()['Account']


def save_endpoint_info(
    input_dict:dict,
    bucket:str,
    output_path:str
    ):
    
    """
    Função para salvar informações sobre o endpoint no diretório S3 especificado.
    
    ---
    
    Variáveis:
    - input_dict (dict): dicionário com informações sobre o endpoint (model_monitor.describe_schedule()). \n
    - bucket (str): bucket para saída da função. \n
    - output_path (str): caminho até o diretório do S3 onde as informações serão salvas. \n
    
    """
    
    endpoint_info = dict()
    endpoint_info['endpoint_name'] = input_dict['EndpointName']
    endpoint_info['model_monitoring_schedule'] = input_dict['MonitoringScheduleName']
    endpoint_info['schedule_arn'] = input_dict['MonitoringScheduleArn']

    key = output_path + '/endpoint_info.json'
    s3 = boto3.client('s3')
    s3.put_object(
         Body=json.dumps(endpoint_info),
         Bucket=bucket,
         Key=key
    )
    
    return


if __name__ == '__main__':
    
    # bucket e endereços dos datasets de treino e validação
    bucket='ons-ds-mlops'
    training_path = 'framework-overview/data/processed/train/dataset.csv'
    validation_path = 'framework-overview/data/processed/validation/dataset.csv'
    trainset_uri=f's3://{bucket}/{training_path}'
    validationset_uri=f's3://{bucket}/{validation_path}'

    # container contendo o modelo
    container = sagemaker.image_uris.retrieve(region=session.region_name, framework='xgboost', version='latest')

    # definição de hiperparametros
    hyperparameters={
        'eta':0.4,
        'gamma':1,
        'max_depth':3,
        'num_round':20,
        'min_child_weight':8,
        'subsample':0.6,
        'objective':"multi:softmax",
        'num_class':5
    }

    # uri para artefatos do modelo
    model_uri = f's3://{bucket}/framework-overview/artifacts/model'

    # treinar o modelo
    model = utils.train_model(
        container=container,
        trainset_uri=trainset_uri,
        validationset_uri=validationset_uri,
        model_uri=model_uri,
        hyperparameters=hyperparameters
        )
    
    # listar arquivos de modelos no bucket
    s3_client = session.client('s3')
    result = s3_client.list_objects(Bucket=bucket, Prefix='framework-overview/artifacts/model')
    files = [file.get('Key') for file in result.get('Contents')]
    files = [f's3://{bucket}/{file}' for file in files if 'model.tar.gz' in file]
    
    # selecionar modelo mais recente como artefato de referência para implantação
    model_path = files[-1]
    print(f'artefatos do modelo de referência em: {model_path}')
    
    # configuração de captura de dados
    data_capture_uri = f's3://{bucket}/framework-overview/data/capture/'
    data_capture_config = utils.get_data_capture_config(
        storage_uri=data_capture_uri,
        sampling_percentage=100
        )
    
    # implantação do modelo em um endpoint
    stage = 'dev'
    timehash = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    endpoint_name = f"predicao-climatica-{stage}-{timehash}"
    print(f'nome do endpoint: {endpoint_name}')
    endpoint = utils.deploy_model(
        container=container,
        model_uri=model_path,
        endpoint_name=endpoint_name,
        data_capture_config=data_capture_config
    )
    
    # declaração do baseline
    baseline_path = f'framework-overview/model-monitor/baseline'
    model_monitor = utils.create_model_monitor_baseline(
        bucket=bucket,
        baseline_path=baseline_path,
        trainset_uri=trainset_uri
        )
    
    # configurar rotina de monitoramento
    schedule_name = f'schedule-mm-predicao-climatica-{timehash}'
    output_monitor_uri = f's3://{bucket}/framework-overview/model-monitor/output'
    model_monitor = utils.create_monitoring_schedule(
        model_monitor,
        endpoint_name,
        monitor_schedule_name=schedule_name,
        output_s3_uri=output_monitor_uri
        )
    
    print('Monitor de modelos configurado com sucesso')
    
    endpoint_info_output_path = 'framework-overview/artifacts/endpoint'
    save_endpoint_info(
        input_dict=model_monitor.describe_schedule(),
        bucket=bucket,
        output_path=endpoint_info_output_path
        )

    print(f'Informações do endpoint salvas em s3://{bucket}/{endpoint_info_output_path}')