import os
import gc
import sys
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split


def get_train_data(
    bucket:str='gpa-churn',
    prefix:str='data/processed/input/after_stix_'
    ):
    
    storage_client = storage.Client()
    obj_list = storage_client.list_blobs(bucket)
    obj_list = [i.name for i in obj_list if prefix in i.name]
    
    df_list = []
    for obj in obj_list:
        local_df = pd.read_parquet('gs://gpa-churn/'+obj)
        df_list.append(local_df)
        print(f'added {prefix}{obj}')

    df = pd.concat(df_list, axis=0)
    df.drop(columns=['cod_cliente'], inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    target = 'target'
    features = list(df.columns)
    features = [i for i in features if i != target]

    Xtrain, Xval, ytrain, yval = train_test_split(
        df[features], 
        df[[target]],
        test_size=0.15, 
        random_state=501
        )
    
    del df
    gc.collect()

    Xtrain.reset_index(drop=True, inplace=True)
    Xval.reset_index(drop=True, inplace=True)
    ytrain.reset_index(drop=True, inplace=True)
    yval.reset_index(drop=True, inplace=True)
    print('Successfully read training data.')
    print('shapes:')
    print(f'xtrain:{Xtrain.shape}, ytrain:{ytrain.shape}')
    print(f'xval:{Xval.shape}, yval:{yval.shape}')
    
    return Xtrain, Xval, ytrain, yval


def get_prediction_data(
    bucket:str='gpa-churn',
    prefix:str='data/processed/test/after_stix_'
    ):
    
    storage_client = storage.Client()
    obj_list = storage_client.list_blobs(bucket)
    obj_list = [i.name for i in obj_list if prefix in i.name]
    
    df_list = []
    for obj in obj_list:
        local_df = pd.read_parquet('gs://gpa-churn/'+obj)
        df_list.append(local_df)
        print(f'added {prefix}{obj}')

    df = pd.concat(df_list, axis=0)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    target = 'target'
    features = list(df.columns)
    features = [i for i in features if i != target]

    X = df[features]
    print('Successfully read test data.')
    print(f'X:{X.shape}')
    
    return X