import pytz
from datetime import datetime as dt

def log(frase):
    time_now = dt.now(tz=pytz.timezone('America/Sao_Paulo')).strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{time_now}] {frase}')
    return



# # for example only
# import boto3
# import botocore
# from botocore import UNSIGNED
# from botocore.config import Config
# from io import BytesIO
# import os
# from glob import glob
# import pandas as pd

# def import_s3_example_dataset(dataset):
#     '''
#     inputs:
#         dataset: 'marketing' or 'bombas'
#     '''
    
#     BUCKET_NAME = 'ds-example-datasets' 
#     available_datasets = {
#         'marketing': 'marketing.feather',
#         'bombas': 'bombas.feather',
#     }
    
    
#     #os.mkdir(path_to_experiment)
#     if 'downloaded_data' not in glob('*'):
#         os.mkdir('downloaded_data')
    
#     if f'downloaded_data/{dataset}.feather' not in glob('downloaded_data/*'):
#         print(f'Baixando dataset {dataset.upper()} do S3...')
        
#         (
#             boto3
#             .resource('s3', config=Config(signature_version=UNSIGNED))
#             .Bucket(BUCKET_NAME)
#             .download_file(available_datasets[dataset], f'downloaded_data/{dataset}.feather')
#         )
        
#     df = pd.read_feather(f'downloaded_data/{dataset}.feather')
    
#     return df