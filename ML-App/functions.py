# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:35:33 2023

@author: gitan
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

# function to convert dataframe to csv
def df_to_csv(df, filename):
     df.to_csv(filename, index=False)

# function to convert dataframe to html
def df_to_html(df, filename):
     df.to_html(filename, index=False)

# function to convert dataframe to excel
def df_to_excel(df, filename):
     df.to_excel(filename, index=False)

def expolaration(dataset):
    data = dataset
    info_data = pd.DataFrame({'col': data.columns, 'type': data.dtypes, 'nulls': data.isnull().sum(), 'unique': data.nunique()})
    info_data2 = info_data.reset_index()
    desc_data_noobject = data.describe().T
    desc_data_object = data.describe(include='O').T
    nunique_data = pd.DataFrame({'unique': data.nunique()})
    unique_data = pd.DataFrame({'unique': data.apply(lambda x: x.unique())})
    data_values = pd.concat([unique_data, nunique_data],axis=1)
    data_values2 = pd.concat([desc_data_noobject, desc_data_object],axis=1)
    data_values2 = data_values2.drop(['unique'],axis=1)
    data_values2 = data_values2.replace(np.nan, 'null')
    data_fig = pd.merge(data_values2,data_values,left_index = True, right_index= True)
    data_fig = data_fig.reset_index()
    data_fig = data_fig.rename(columns={'index': 'column'})
    data_fig = pd.merge(data_fig,info_data2['type'],left_index = True, right_index= True)
    data_fig = pd.merge(data_fig,info_data2['nulls'],left_index = True, right_index= True)
    df_to_csv(data_fig,'data_explorations.csv')
    df_to_excel(data_fig,'data_explorations.xlsx')
    return data_fig

def factorize(dataset, columns):
    for i in columns:
        dataset[i], _ = pd.factorize(dataset[i])


def where(dataset, columns):
    for i in columns:
        dataset[i],_ = np.where((dataset[i] == 'Yes'),1,0)
        
        

class WhereTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = np.where((X_copy[col] == 'Yes'), 1, 0)
        return X_copy


# class MostFrequentImputer(TransformerMixin):
#     def __init__(self, target_col):
#         self.target_col = target_col
        
#     def fit(self, X, y=None):
#         self.modes_ = {}
#         for col in X.columns:
#             if col != self.target_col:
#                 self.modes_[col] = X.groupby(self.target_col)[col].agg(lambda x: x.mode().iloc[0]).to_dict()
#         return self
    
#     def transform(self, X):
#         X_copy = X.copy()
#         for col in X.columns:
#             if col != self.target_col:
#                 X_copy[col] = X_copy.apply(lambda row: self.modes_[col][row[self.target_col]]
#                                            if pd.isna(row[col]) else row[col], axis=1)
#         return X_copy
    
#     def get_params(self, deep=True):
#         return {'target_col': self.target_col}

class MostFrequentImputer(TransformerMixin):
    def __init__(self):
        self.fill = None

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == 'O' else X[c].mean() for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    def get_params(self, deep=True):
        return {}
    
    
    
    
    
    
    
    