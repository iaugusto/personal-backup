import os 
import numpy as np
import pandas as pd
import seaborn as sn 
from pathlib import Path
from datetime import datetime


class feature_selection():


    #  --------------------
    def __init__(self):
        self.gen_date = datetime.now()
        self.dir = os.getcwd()
        return
    #  --------------------


    #  --------------------
    def recursive_correlation_term(
        self,
        dataset:pd.DataFrame,
        threshold:float=0.8
        ) -> pd.DataFrame:

        grouped_features = self.intersec_correlation_definition(
            dataset=dataset,
            threshold=threshold
            )

        if not grouped_features:
            return [], False
        else: 
            return grouped_features[0], True

    #  --------------------


    #  --------------------
    def bruteforce_correlation_definition(
        self,
        dataset:pd.DataFrame,
        threshold:float=0.8
        ) -> pd.DataFrame:

        var_corr = set()
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i,j]) > threshold:
                    col = corr_matrix.columns[i]
                    var_corr.add(col)

        return var_corr
    #  --------------------
    
    
    #  --------------------
    def intersec_correlation_definition(
        self,
        dataset:pd.DataFrame,
        threshold:float=0.8
        ) -> pd.DataFrame:

        var_corr = dataset.corr()
        var_corr = var_corr.abs().unstack()
        var_corr = var_corr.sort_values(ascending=False)
        var_corr = var_corr[var_corr>= threshold]
        var_corr = pd.DataFrame(var_corr).reset_index()
        var_corr.columns = ['feature_1', 'feature_2', 'corr']
        var_corr = var_corr[var_corr['feature_1']!=var_corr['feature_2']]

        grouped_features = []

        for feature in var_corr['feature_1'].unique():
            corr_features = []
            if feature not in grouped_features:
                correlated_block = var_corr[var_corr['feature_1']==feature]
                corr_features = corr_features + \
                    list(correlated_block['feature_2'].unique()) + [feature]
            grouped_features.append(corr_features)

        return grouped_features
    #  --------------------