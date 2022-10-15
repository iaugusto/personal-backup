import io
import re
import os
import sys
import json
import spacy
import boto3
import random
import shutil
import sagemaker
import subprocess
import numpy as np
import pandas as pd
import awswrangler as wr
from datetime import datetime
from spacy.scorer import Scorer
from spacy.training import Example
from sagemaker import get_execution_role
from spacy.util import minibatch, compounding

session = boto3.Session()
sm = session.client('sagemaker', region_name='us-east-1')
role = 'arn:aws:iam::478704051461:role/service-role/AmazonSageMaker-ExecutionRole-20220317T121940' #sagemaker.get_execution_role()
account_id = session.client('sts').get_caller_identity()['Account']


def save_sample_to_s3(
    X,
    bucket:str,
    filepath:str
    ):

    sample_path = f's3://{bucket}/{filepath}'
    X.to_parquet(sample_path, compression='gzip')
    print(f'saved samples in {sample_path}')
    
    return


def get_samples(
    bucket:str='ons-ds-dev-entityrecognition',
    path:str='mpo/mvp/data/raw/incremental/predefined/'
    ):

    import boto3
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(bucket)
    df = pd.DataFrame()

    file_list = []
    for file in my_bucket.objects.all():
        file_list.append(file.key)
    file_list = [i for i in file_list if path in i]
    file_list = [i for i in file_list if '.parquet' in i]

    if file_list:
        df_list = []
        uri_prefix = f's3://{bucket}/'
        for file_path in file_list:
            df_local = pd.read_parquet(uri_prefix+file_path)
            df_local = df_local.drop_duplicates()
            df_local = df_local.dropna()
            df_list.append(df_local)

        df = pd.concat(df_list)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


def create_trainset(
    bucket:str,
    processed_path:str='mpo/mvp/data/processed',
    incremental_path:str='mpo/mvp/data/incremental',
    predefined_output_path:str='mpo/mvp/data/train/predefined/dataset.parquet',
    inequalities_output_path:str='mpo/mvp/data/train/inequalities/dataset.parquet'
    ):
    
    predefined_processed_path = f'{processed_path}/predefined/'
    predefined_incremental_path = f'{incremental_path}/predefined/'
    inequalities_processed_path = f'{processed_path}/inequalities/'
    inequalities_incremental_path = f'{incremental_path}/inequalities/'
    
    print('-'*10)
    print('Creating trainset')
    
    samples = get_samples(
        bucket=bucket,
        path=predefined_processed_path
    )
    print('samples for predefined model collected')
    samples_incremental = get_samples(
        bucket=bucket,
        path=predefined_incremental_path
    )
    if len(samples_incremental) > 0:
        samples = pd.concat([samples, samples_incremental], axis=0)
        #samples = samples.drop_duplicates()
        print('incremental samples for predefined model collected ')
    else:
        print('no incremental samples found for predefined model')
    
    save_sample_to_s3(
        samples,
        bucket=bucket,
        filepath=predefined_output_path
    )
    print(f'predefined samples saved to bucket s3://{bucket}/{predefined_output_path}.')
    
    samples_inequalities = get_samples(
        bucket=bucket,
        path=inequalities_processed_path
    )
    print('samples for inequalities model collected')
    
    samples_inequalities_incremental = get_samples(
        bucket=bucket,
        path=inequalities_incremental_path
    )
    if len(samples_inequalities_incremental) > 0:
        samples_inequalities = pd.concat([samples_inequalities, samples_inequalities_incremental], axis=0)
        #samples_inequalities = samples_inequalities.drop_duplicates()
        print('incremental samples for inequalities model collected ')
    else:
        print('no incremental samples found for inequalities model')
    
    save_sample_to_s3(
        samples_inequalities,
        bucket=bucket,
        filepath=inequalities_output_path
    )
    print(f'inequalities samples saved to bucket s3://{bucket}/{inequalities_output_path}.')
    
    return


def get_input_data(
    bucket:str='ons-ds-dev-entityrecognition',
    path:str='mpo/mvp/data/train/predefined/dataset.parquet'
    ):
    
    """
    Função para leitura dos dados processados no formato .parquet.
    A função segue a formatação (s3://{bucket}/{path}) para o uri do objeto.
    
    ---
    
    Variáveis:
    - bucket (str): nome do bucket. \n
    - path (str): caminho até o dataset de treino. \n
    
    """
    
    train_dataset = wr.s3.read_parquet(f's3://{bucket}/{path}')
    train_dataset.sort_values(by=['text','start', 'end'], inplace=True)
    train_dataset[['start','end']] = train_dataset[['start','end']].astype('int')
    
    return train_dataset


def prepare_input_data(
    train_dataset
    ):
    
    """
    Função para preparação/formatação do dataset segundo requerimentos do SpaCy. 
    
    ---
    
    Variáveis:
    - train_dataset (pd.DataFrame): dataset com dados de treinamento. \n
    
    """
    
    training_set = []
    unique_text = sorted(set(train_dataset['text']))
    for samp in unique_text:
        df_loop = train_dataset[train_dataset['text']==samp]
        df_loop.drop_duplicates(inplace=True)
        df_loop.sort_values(by=['start','end'], axis=0, ascending=True, inplace=True)
        sliding_ref = -1
        ent_list = []
        for index, row in df_loop.iterrows():
            if row['start'] > sliding_ref:
                ent_list.append([row['start'], row['end'], row['label']])
                sliding_ref = row['end']
        training_set.append([samp, {'entities': tuple(ent_list)}])
    
    return training_set


def create_spacy_model(
    blank:bool=True
    ):
    
    """
    Função para criar um modelo de reconhecimento de entidades do SpaCy. 
    A função retorna o modelo inicializado/carregado.
    
    """
    
    if blank:
        nlp = spacy.blank("pt")
    else:
        nlp = spacy.load("pt_core_news_sm")
    
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    return nlp


def add_labels_to_ner(
    nlp,
    label_list:list
    ):
    
    """
    Função para adicionar entidades ao SpaCy NER.
    
    ---
    
    Variáveis:
    - nlp (spacy model) : modelo do spacy contendo um módulo de NER. \n
    - label_list (list) : lista com labels a serem adicionados ao NER. \n
    
    """
    
    ner = nlp.get_pipe("ner")
    for label in label_list:#set(train_dataset['label'].to_list()):
        ner.add_label(label)
    
    return nlp


def train_ner_model(
    nlp,
    training_set,
    n_iterations:int=40,
    batch_size:int=50,
    drop_rate:int=0.4
    ):
    
    """
    Função para adicionar entidades ao SpaCy NER.
    
    ---
    
    Variáveis:
    - nlp (spacy model) : modelo do spacy contendo um módulo de NER. \n
    - training_set (list) : lista com texto, e labels marcados. \n
    
    """
    
    import warnings
    
    warnings.simplefilter("ignore")
    loss_list = []
    pipes_except_ner = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    
    if pipes_except_ner:
        print('There are other pipelines except Named Entity Recognition in Spacy module')
    else:
        print('There are no pipelines except Named Entity Recognition in Spacy module')
        nlp.begin_training()
    
    optimizer = nlp.create_optimizer()
    
    # only training NER module - no other module can be retrained in this section
    print('-'*10)
    print('training sequence')
    with nlp.disable_pipes(*pipes_except_ner):
        for it in range(n_iterations):
            random.shuffle(training_set)
            losses = {}       
            batches = minibatch(
                training_set,
                size=batch_size
                )
            for batch in batches:
                texts, annotations = zip(*batch)
                for text, annot in zip(texts, annotations):
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annot)
                    #spacy.training.offsets_to_biluo_tags(doc, annot['entities'])
                    nlp.update(
                        [example],
                        sgd=optimizer, 
                        drop=drop_rate, 
                        losses=losses
                        )
            print('iterations: ', it+1, 'Loss: ', losses)
            loss_list.append(losses['ner'])
    print('-'*10)
    
    return nlp


def load_ner_model(
    bucket:str='ons-ds-dev-entityrecognition',
    path:str='mpo/mvp/artifacts/model/predefined',
    model_id:str='entity-recognition-model-2022-08-29-18-07-02'
    ):
    
    """
    Função para leitura dos dados processados no formato .parquet.
    A função segue a formatação (s3://{bucket}/{path}) para o uri do objeto.
    
    ---
    
    Variáveis:
    - bucket (str): nome do bucket. \n
    - path (str): caminho até o dataset de treino. \n
    
    """
    
    from zipfile import ZipFile
    
    s3 = boto3.resource('s3')
    file_path = f'{path}/{model_id}.zip'
    print(file_path)
    s3.Bucket(bucket).download_file(file_path, model_id+'.zip')
    
    with ZipFile(model_id+'.zip', 'r') as zipObj:
        zipObj.extractall(model_id)
        
    nlp = spacy.load(model_id)
    
    os.remove(model_id+'.zip')
    shutil.rmtree(f'{model_id}', ignore_errors=True)
    
    return nlp


def save_model_to_s3(
    nlp,
    bucket:str='ons-ds-dev-entityrecognition',
    path:str='mpo/mvp/artifacts/model/predefined'
    ):
    
    nlp.to_disk('model')
    model_id = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    shutil.make_archive(model_id, 'zip', 'model')
    
    file_path = f'{path}/{model_id}.zip'
    s3 = boto3.resource('s3')
    s3.Bucket(bucket).upload_file(f'{model_id}.zip', file_path)
    
    return model_id


def create_metrics_dict(
    nlp, 
    training_set
    ):
    
    examples = []
    scorer = Scorer()
    for text, annotations in training_set:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        example.predicted = nlp(str(example.predicted))
        examples.append(example)

    score_result_se = scorer.score(examples)
    metrics_dict = score_result_se['ents_per_type']
    
    return metrics_dict


def insert_new_metrics(
    metrics_dict,
    bucket:str='ons-ds-dev-entityrecognition',
    path:str='mpo/mvp/artifacts/model_info/predefined/metrics.parquet',
    model_id:str='model-default',
    baseline_id:str='2022-09-22-14-41-00'
    ):
    
    metrics = pd.DataFrame(metrics_dict)
    metrics = pd.DataFrame(metrics.mean(axis=1), columns=['metrics']).transpose()
    metrics['variant'] = model_id
    metrics['baseline'] = baseline_id
    obj_uri = f's3://{bucket}/{path}'
    
    try:
        s3_metrics = pd.read_parquet(obj_uri)
        s3_metrics = pd.concat([s3_metrics, metrics], axis=0)
    except:
        s3_metrics = metrics
    
    s3_metrics.drop_duplicates(keep='first', inplace=True)
    s3_metrics.sort_values(by=['baseline','variant'], inplace=True)
    s3_metrics.to_parquet(obj_uri)
    
    return s3_metrics


def load_metrics_from_s3(
    bucket:str='ons-ds-dev-entityrecognition',
    path:str='mpo/mvp/artifacts/model_info/predefined/metrics.parquet'
    ):
    
    obj_uri = f's3://{bucket}/{path}'
    metrics = pd.read_parquet(obj_uri)
    
    return metrics


def upload_image_to_ecr():
    
    commands = [
        'cd ../',
        'sm-docker build . --repository ons-ds-entity-recognition:latest'
    ]
    joint_commands = ';'.join(commands)
    result = subprocess.run(joint_commands, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    return result


def create_sagemaker_model(
    model_id,
    stage:str='dev'
    ):
    
    account_id = boto3.client('sts').get_caller_identity()['Account']
    sm_client = boto3.client(service_name='sagemaker')
    region = boto3.Session().region_name
    role = get_execution_role()
    
    model_name = f'mpo-ner-{stage}-{model_id}'
    container = '{}.dkr.ecr.{}.amazonaws.com/ons-ds-entity-recognition:latest'.format(account_id, region)

    print('Model name: ' + model_name)
    print('Container image: ' + container)

    container = {
        'Image': container
    }

    create_model_response = sm_client.create_model(
        ModelName = model_name,
        ExecutionRoleArn = role,
        Containers = [container])
    
    print("Model Arn: " + create_model_response['ModelArn'])
    
    return model_name


def clean_sagemaker_environment():
    
    session = boto3.Session()
    sm = session.client('sagemaker')
    client = boto3.client('sagemaker')
    
    registered_ner_models = sm.list_models()['Models']
    registered_ner_models = [i['ModelName'] for i in registered_ner_models if 'mpo-ner' in i['ModelName']]
    
    if registered_ner_models[3:]:
        for model_name in registered_ner_models[3:]:
            client.delete_model(ModelName=model_name)
            
    return


def create_endpoint_configuration(
    model_id:str,
    stage:str='dev'
    ):
    
    model_name = f'mpo-ner-{stage}-{model_id}'
    
    sm_client = boto3.client(service_name='sagemaker')
    instance_type = 'ml.t2.large'

    endpoint_config_name = f'mpo-ner-{stage}-endpoint-config'
    print('Endpoint config name: ' + endpoint_config_name)
    
    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName = endpoint_config_name,
        ProductionVariants=[{
            'InstanceType': instance_type,
            'InitialInstanceCount': 1,
            'InitialVariantWeight': 1,
            'ModelName': model_name,
            'VariantName': 'AllTraffic'}])

    print("Endpoint config Arn: " + create_endpoint_config_response['EndpointConfigArn'])
    
    return endpoint_config_name


def deploy_model_as_endpoint(
    endpoint_config_name,
    stage:str='dev'
    ):
    
    sm_client = boto3.client(service_name='sagemaker')
    
    endpoint_name = f'mpo-ner-{stage}-endpoint'
    print('Endpoint name: ' + endpoint_name)
    
    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name)
    print('Endpoint Arn: ' + create_endpoint_response['EndpointArn'])
    
    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp['EndpointStatus']
    print("Endpoint Status: " + status)

    print('Waiting for {} endpoint to be in service...'.format(endpoint_name))
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    
    return endpoint_name


def update_endpoint(
    endpoint_config_name,
    stage:str='dev'
    ):
    
    sm_client = boto3.client(service_name='sagemaker')
    
    endpoint_name = f'mpo-ner-{stage}-endpoint'
    print('Endpoint name: ' + endpoint_name)
    
    create_endpoint_response = sm_client.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name)
    print('Endpoint Arn: ' + create_endpoint_response['EndpointArn'])
    
    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp['EndpointStatus']
    print("Endpoint Status: " + status)

    print('Waiting for {} endpoint to be in service...'.format(endpoint_name))
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    
    return endpoint_name


def register_production_variant(
    endpoint_name:str,
    model_id:str='model-default',
    bucket:str='ons-ds-dev-entityrecognition',
    path:str='mpo/mvp/artifacts/production_variant/predefined/production.parquet'
    ):
    
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    production_variant_info = {
        'model':model_id,
        'endpoint':endpoint_name,
        'rollout':now_str,
    }
    production_variant_df = pd.DataFrame(production_variant_info, index=[0])
    obj_uri = f's3://{bucket}/{path}'
    
    try:
        s3_production_variant_df = pd.read_parquet(obj_uri)
        s3_production_variant_df = pd.concat([s3_production_variant_df, production_variant_df], axis=0)
    except:
        s3_production_variant_df = production_variant_df
    
    production_variant_df.drop_duplicates(keep='first', inplace=True)
    production_variant_df.sort_values(by=['rollout'], ascending=False, inplace=True)
    production_variant_df.reset_index(drop=True, inplace=True)
    production_variant_df.to_parquet(obj_uri)
    
    return production_variant_df


def load_production_variant(
    bucket:str='ons-ds-dev-entityrecognition',
    path:str='mpo/mvp/artifacts/production_variant/predefined/production.parquet'
    ):
    
    obj_uri = f's3://{bucket}/{path}'
    s3_production_variant_df = pd.read_parquet(obj_uri)
    s3_production_variant_df.sort_values(
        by=['rollout'], 
        ascending=False, 
        inplace=True
    )
    production_variant = s3_production_variant_df['model'].iloc[0]

    return production_variant


def delete_model(
    model_id:str,
    stage:str='dev'
    ):
    
    """
    Função para deletar um modelo.
    
    ---
    
    Variáveis:
    - model_id (str): label designado ao modelo. \n
    """
    
    model_name = f'mpo-ner-{stage}-{model_id}'
    sm.delete_model(ModelName=model_name)
    
    print(f'Modelo {model_id} excluído com sucesso')
    
    return


def delete_endpoint_configuration(
    stage:str='dev'
    ):
    
    """
    Função para deletar uma configuração de endpoint.
    
    ---
    
    Variáveis:
    - model_id (str): label designado à configuração de endpoint. \n
    """
    
    endpoint_config_name = f'mpo-ner-{stage}-endpoint-config'
    sm.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    
    print(f'Configuração de endpoint {endpoint_config_name} excluída com sucesso')
    
    return


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


def load_latest_baseline(
    bucket:str='ons-ds-dev-entityrecognition',
    path:str='mpo/mvp/artifacts/baseline/predefined'
    ):

    latest_baseline = pd.DataFrame()
    baseline_id = None
    
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(bucket)
    s3_client = boto3.client('s3')

    file_list = []
    for file in my_bucket.objects.all():
        file_list.append(file.key)
    file_list = [i for i in file_list if path in i]
    file_list = [i for i in file_list if '.parquet' in i]
    
    try:
        latest_baseline_file = sorted(file_list)[-1]
        baseline_id = latest_baseline_file.split('/')[-1]
        baseline_id = baseline_id.split('.')[0]
        uri_prefix = f's3://{bucket}/'
        obj_uri = uri_prefix+latest_baseline_file
        obj = s3_client.get_object(Bucket=bucket, Key=latest_baseline_file)
        latest_baseline = pd.read_parquet(io.BytesIO(obj['Body'].read()))
        print('found latest baseline: ', baseline_id)
    except:
        print('there is no previous baseline.')
    
    return latest_baseline, baseline_id


def update_baseline(
    train_dataset,
    bucket:str='ons-ds-dev-entityrecognition',
    path:str='mpo/mvp/artifacts/baseline/predefined'
    ):
    
    print('loading current baseline')
    latest_baseline, baseline_id = load_latest_baseline(
        bucket=bucket,
        path=path
        )
    
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    obj_uri = f's3://{bucket}/{path}/{timestamp}.parquet'
    print('checking status')
    training_status = True
    if baseline_id:
        if latest_baseline.ne(train_dataset).any().any() == False:
            training_status = False
        else:
            train_dataset.to_parquet(obj_uri, compression='gzip')
    else:
        train_dataset.to_parquet(obj_uri, compression='gzip')
    
    print('updated latest baseline')
    latest_baseline, baseline_id = load_latest_baseline(
        bucket=bucket,
        path=path
        )
    
    return training_status, baseline_id


def process_workload(
    bucket:str='ons-ds-dev-entityrecognition',
    model:str='predefined'
    ):
    
    print('-'*10)
    print('Processing new workload')
    
    trainset_path=f'mpo/mvp/data/train/{model}/dataset.parquet'
    train_dataset = get_input_data(
        bucket=bucket,
        path=trainset_path
        )
    
    path = f'mpo/mvp/artifacts/baseline/{model}'
    training_status, baseline_id = update_baseline(
        train_dataset=train_dataset,
        bucket=bucket,
        path=path
        )
    
    return training_status, baseline_id


def run_training_pipeline(
    bucket:str='ons-ds-dev-entityrecognition',
    model:str='predefined',
    train_from_blank:bool=True,
    baseline_id:str='2022-09-22-14-41-00'
    ):
    
    print('-'*10)
    print('Running training pipeline')
    
    bool_translation_dict = {
        'predefined':True,
        'inequalities':False
    }
    
    production_variant_path=f'mpo/mvp/artifacts/production_variant/{model}/production.parquet'
    metrics_path=f'mpo/mvp/artifacts/model_info/{model}/metrics.parquet'
    trainset_path=f'mpo/mvp/data/train/{model}/dataset.parquet'
    model_path=f'mpo/mvp/artifacts/model/{model}'
    blank = bool_translation_dict[model]
    
    if model == 'inequalities':
        n_iterations=20
        batch_size=30
        drop_rate=0.35
    else:
        n_iterations=45
        batch_size=50
        drop_rate=0.40
    
    train_dataset = get_input_data(
        bucket=bucket,
        path=trainset_path
        )
    print('loaded train_dataset from processed s3 directory')
    
    training_set = prepare_input_data(
        train_dataset
        )
    print('created trainset (annotations) from train_dataset (pd.DataFrame)')
    
    if train_from_blank:
        print('training new spacy model - from blank pt model')
        nlp = create_spacy_model(blank=blank)
    else:
        print('training over previous production variant')
        try:
            production_variant = load_production_variant(
                bucket=bucket,
                path=production_variant_path
            )
            print(f'production variant {production_variant}')
            nlp = load_ner_model(
                bucket=bucket,
                path=model_path,
                model_id=production_variant
                )
            print('loaded named entity recognition model from production')
        except:
            print('it was not possible to iterate over the  production variant')
            print('creating new spacy model - from blank pt model')
            nlp = create_spacy_model(blank=blank)
    
    nlp = train_ner_model(
        nlp,
        training_set,
        n_iterations=n_iterations,
        batch_size=batch_size,
        drop_rate=drop_rate
        )
    print('trained/updated custom entity recognition model')
    
    model_id = save_model_to_s3(
        nlp,
        bucket=bucket,
        path=model_path
        )
    print('saved model to s3')
    
    metrics_dict = create_metrics_dict(
        nlp, 
        training_set
        )
    print('created metrics')
    
    metrics = insert_new_metrics(
        metrics_dict,
        bucket=bucket,
        path=metrics_path,
        model_id=model_id,
        baseline_id=baseline_id
        )
    print('Inserted metrics to model registry')
    
    return model_id, metrics


def get_deployed_metrics_with_new_baseline(
    model:str='predefined',
    bucket:str='ons-ds-dev-entityrecognition',
    baseline_id:str='2022-09-22-14-41-00'
    ):
    
    print('-'*10)
    print('Updating deployed model\'s metrics with new data baseline')
    
    production_variant_path=f'mpo/mvp/artifacts/production_variant/{model}/production.parquet'
    metrics_path=f'mpo/mvp/artifacts/model_info/{model}/metrics.parquet'
    trainset_path=f'mpo/mvp/data/train/{model}/dataset.parquet'
    model_path=f'mpo/mvp/artifacts/model/{model}'
    
    train_dataset = get_input_data(
        bucket=bucket,
        path=trainset_path
        )
    
    training_set = prepare_input_data(
        train_dataset
        )
    
    production_variant = load_production_variant(
        bucket=bucket,
        path=production_variant_path
        )
    
    nlp = load_ner_model(
        bucket=bucket,
        path=model_path,
        model_id=production_variant
        )
    
    metrics_dict = create_metrics_dict(
        nlp, 
        training_set
        )
    
    metrics = insert_new_metrics(
        metrics_dict,
        bucket=bucket,
        path=metrics_path,
        model_id=production_variant,
        baseline_id=baseline_id
        )
    
    print('Updated production metrics updated with new data baseline to model registry')
    
    return production_variant, metrics


def get_production_variant(
    metrics
    ):
    
    latest_baseline = sorted(metrics['baseline'])[-1]
    latest_metrics = metrics[metrics['baseline']==latest_baseline]
    latest_metrics = latest_metrics.sort_values(by=['f','r','p'], ascending=False)
    production_variant = latest_metrics['variant'].iloc[0]
    print(f'production variant selected: {production_variant}')
    
    return production_variant


def check_production_variant_availability(
    production_variant:str,
    bucket:str='ons-ds-dev-entityrecognition',
    path:str='mpo/mvp/artifacts/production_variant/predefined/production.parquet'
    ):
    
    status = False
    obj_uri = f's3://{bucket}/{path}'
    try:
        production_variant_df = wr.s3.read_parquet(obj_uri)
        production_variant_df.sort_values(by=['rollout'], ascending=False, inplace=True)
        deployed_variant = production_variant_df['model'].iloc[0]
        if production_variant == deployed_variant:
            status = True
    except:
        status = status
    
    return status


def prepare_production_variant(
    production_variant:str,
    model:str='predefined',
    bucket:str='ons-ds-dev-entityrecognition'
    ):
    
    model_path=f'mpo/mvp/artifacts/model/{model}'
    nlp = load_ner_model(
        bucket=bucket,
        path=model_path,
        model_id=production_variant
        )
    print(f'loaded {production_variant}')
    
    artifact_path = f'../artifacts/{model}'
    nlp.to_disk(artifact_path)
    
    return


def deployment_sequence(
    predefined_id:str='default',
    inequalities_id:str='default',
    bucket:str='ons-ds-dev-entityrecognition',
    stage:str='dev'
    ):
    
    print('-'*10)
    print('Running deployment sequence')
    
    model_name = f'{predefined_id}-{inequalities_id}'
    
    result = upload_image_to_ecr()
    print('uploaded image to ECR')
        
    try:
        sm_model_name = create_sagemaker_model(
            model_id=model_name,
            stage=stage
        )
        print(f'uploaded sagemaker model: {sm_model_name}')
    except:
        delete_model(
            model_id=model_name,
            stage=stage
        )
        sm_model_name = create_sagemaker_model(
            model_id=model_name,
            stage=stage
        )
        print(f'subscribed sagemaker model: {sm_model_name}')
        
    try:
        endpoint_config_name = create_endpoint_configuration(
            model_id=model_name,
            stage=stage
        )
        print(f'uploaded sagemaker endpoint configuration: {endpoint_config_name}')
    except:
        delete_endpoint_configuration(
            stage=stage
            )
        endpoint_config_name = create_endpoint_configuration(
            model_id=model_name,
            stage=stage
        )
        print(f'subscribed sagemaker endpoint configuration: {endpoint_config_name}')
            
    try:
        endpoint_name = deploy_model_as_endpoint(
            endpoint_config_name=endpoint_config_name,
            stage=stage
            )
        print(f'deployed endpoint: {endpoint_name}')
    except:
        endpoint_name = update_endpoint(
            endpoint_config_name,
            stage=stage
            )
        print(f'updated endpoint: {endpoint_name}')
        
    predefined_prod_df = register_production_variant(
        endpoint_name=endpoint_name,
        model_id=predefined_id,
        bucket=bucket,
        path='mpo/mvp/artifacts/production_variant/predefined/production.parquet'
        )
    
    inequalities_prod_df = register_production_variant(
        endpoint_name=endpoint_name,
        model_id=inequalities_id,
        bucket=bucket,
        path='mpo/mvp/artifacts/production_variant/inequalities/production.parquet'
        )
    
    print('deployed new production variant')
    print('predefined labels:')
    print(predefined_prod_df.head())
    print('inequalities:')
    print(inequalities_prod_df.head())
    
    clean_sagemaker_environment()
    print('cleaned sagemaker model registry')
    
    return


# modify entire __main__ code
if __name__=='__main__':
    
    base_dir = 'opt/ml/processing'
    predefined_training_dir = f'{base_dir}/predefined_training'
    Path(predefined_training_dir).mkdir(parents=True, exist_ok=True)
    
    stage = 'hom'
    bucket = f'ons-ds-{stage}-entityrecognition'
    predefined_status = True
    inequalities_status = True
    
    #-------------------------------------------------------------------------
    #loading predefined information
    predefined_workload_path = f'{base_dir}/predefined_workload/predefined.json'
    
    with open(predefined_workload_path, "r") as f:
        predefined_workload_dict = json.load(f)
        
    predefined_training_status = predefined_workload_dict['training_status']
    predefined_baseline_id = predefined_workload_dict['baseline_id']
    
    if predefined_training_status:
        
        predefined_production_variant, predefined_metrics = get_deployed_metrics_with_new_baseline(
            model='predefined',
            bucket=bucket,
            baseline_id=predefined_baseline_id
            )
    
        predefined_model_id, predefined_metrics = run_training_pipeline(
            bucket=bucket,
            model='predefined',
            train_from_blank=True,
            baseline_id=predefined_baseline_id
            )

        predefined_production_variant = get_production_variant(
            predefined_metrics
            )

        predefined_status = check_production_variant_availability(
            production_variant=predefined_production_variant
            )
    
    predefined_deployment_dict = {
        'deployment_status':predefined_status,
        'production_variantt_id':predefined_production_variant
    }
    
    predefined_training_path = f'{predefined_training_dir}/predefined.json'
    
    with open(predefined_training_path, "w") as f:
        f.write(json.dumps(predefined_deployment_dict))