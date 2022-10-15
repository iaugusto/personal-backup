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

import utils

session = boto3.Session()
sm = session.client('sagemaker')
role = sagemaker.get_execution_role()
account_id = session.client('sts').get_caller_identity()['Account']
print(sm, '\n', role, '\n', account_id)


if __name__ == '__main__':
    
    endpoint_name = 'predicao-climatica-dev-2022-07-31-20-50-52'
    bucket='ons-ds-mlops'
    data_path = 'framework-overview/data/processed/validation/dataset.csv'
    data_uri = f's3://{bucket}/{data_path}'
    
    features = validation_df[validation_df.columns[1:]].copy()
    
    output = utils.predict(
        input_data=features,
        endpoint_name=endpoint_name
    )
    
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output.to_csv(f's3://{bucket}/framework-overview/data/output/prediction_{timestamp}.csv', header=False, index=False)