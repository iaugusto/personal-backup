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

df_manualactions = preproc.get_manual_actions(
    bucket='ons-ds-dev-entityrecognition',
    path='mpo/mvp/data/raw/manual_actions'
    )
df_manualactions = preproc.preprocess_fragments(df_manualactions, 'actions')
print('Preprocessed manual actions')

text = text[['processed']].copy()
text = preproc.find_manual_actions(
    text,
    df_manualactions
    )
print('Found manual actions in fragments\' text')

samples = preproc.create_synthetic_samples(
    text,
    entity_col='ACAO_MANUAL',
    text_col='processed'
    )
samples.drop_duplicates(inplace=True)
print('Created samples dataframe for manual actions entity')

bucket = 'ons-ds-dev-entityrecognition'
samples_path = f's3://{bucket}/mpo/mvp/data/processed/manualactions_samples.parquet'
samples.to_parquet(samples_path, compression='gzip')
print(f'saved samples in {samples_path}')