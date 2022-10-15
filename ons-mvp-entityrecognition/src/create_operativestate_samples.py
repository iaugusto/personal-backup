import os
import sys
import numpy as np
import pandas as pd
import awswrangler as wr
from preprocess import preprocess_module


preproc = preprocess_module()
print('Loaded preproc module')

df_fragments = preproc.get_fragments()
text = df_fragments[['Texto Limpo']].drop_duplicates()
text = preproc.preprocess_fragments(text, 'Texto Limpo')
print('Preprocessed fragments')

text = text[['processed']].copy()
text = preproc.find_operativestate(df=text, col='processed')
print('Found operative state in fragments\' text')

samples = preproc.create_synthetic_samples(
    text,
    entity_col='ESTADO_OPERATIVO',
    text_col='processed'
    )
samples.drop_duplicates(inplace=True)
print('Created samples dataframe for operative state entity')

bucket = 'ons-ds-dev-entityrecognition'
samples_path = f's3://{bucket}/mpo/mvp/data/processed/operativestate_samples.parquet'
samples.to_parquet(samples_path, compression='gzip')
print(f'saved samples in {samples_path}')