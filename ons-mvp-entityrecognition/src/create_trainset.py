import os
import sys
import numpy as np
import pandas as pd
import awswrangler as wr
from preprocess import preprocess_module


bucket = 'ons-ds-dev-entityrecognition'

preproc = preprocess_module()
print('Loaded preproc module')

samples = preproc.get_processed_samples()
print('created samples dataframe')
samples_path = f's3://{bucket}/mpo/mvp/data/train/predefined/dataset.parquet'
samples.to_parquet(samples_path, compression='gzip')
print(f'saved samples in {samples_path}')

inequalities_samples = preproc.get_processed_samples_inequalities()
print('created inequalities samples dataframe')
inequalities_samples_path = f's3://{bucket}/mpo/mvp/data/train/inequalities/dataset.parquet'
inequalities_samples.to_parquet(inequalities_samples_path, compression='gzip')
print(f'saved samples in {inequalities_samples_path}')