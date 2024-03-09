# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:00:55 2023

@author: gitan
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from functions import df_to_html
from functions import df_to_csv
from functions import df_to_excel
from functions import expolaration
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from functions import WhereTransformer
from imblearn.over_sampling import SMOTE
from functions import MostFrequentImputer
import joblib
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc

#loading the dataset
data = pd.read_csv("C:/Users/gitan/College/sem4/SupervisedLearning/project/KSI.csv")


### data processing and graphs
dataset = data.drop(['INDEX_','ACCNUM','YEAR','DATE','OFFSET','LATITUDE','LONGITUDE','WARDNUM',
                     'ACCLOC','FATAL_NO',
                 'INITDIR','VEHTYPE','MANOEUVER',
                 'PEDTYPE','PEDACT','PEDCOND',
                 'CYCLISTYPE','CYCACT','CYCCOND',
                 'NEIGHBOURHOOD_158','X','Y',
                 'IMPACTYPE','ACCLASS','INVTYPE',
                 'STREET1','STREET2','NEIGHBOURHOOD_140',
                 'HOOD_140','DIVISION','ObjectId','HOOD_158'], axis = 1)


dataset.columns

dataset.nunique()

data_exp = expolaration(data)

dataset_exp_a = expolaration(dataset)


for i in dataset.columns:
    print("------------------------------")
    print(i)    
    print(dataset[i].unique())

mapping_Road_class = {
    'Major Arterial': 'Arterial',
    'Local': 'Local',
    'Minor Arterial': 'Arterial',
    'Collector': 'Collector',
    'Other': 'other',
    'Pending': 'other',
    'Laneway': 'other',
    'Expressway': 'Expressway',
    'Expressway Ramp': 'Expressway',
    'Major Arterial Ramp': 'Arterial'
}
dataset['ROAD_CLASS'] = dataset['ROAD_CLASS'].replace(mapping_Road_class)

mapping_LOCCOORD = {
    'Intersection': 'at Intersection',
    'Mid-Block': 'not at intersection',
    'Exit Ramp Westbound': 'not at intersection',
    'Exit Ramp Southbound': 'not at intersection',
    'Mid-Block (Abnormal)': 'not at intersection',
    'Entrance Ramp Westbound': 'not at intersection',
    'Park, Private Property, Public Lane': 'not at intersection'
}

dataset['LOCCOORD'] = dataset['LOCCOORD'].replace(mapping_LOCCOORD)

mapping_TRAFFCTL = {
    'No Control': 'no control',
    'Stop Sign': 'control',
    'Traffic Signal': 'control',
    'Pedestrian Crossover': 'control',
    'Traffic Controller': 'control',
    'Yield Sign': 'control',
    'School Guard': 'control',
    'Traffic Gate': 'control',
    'Police Control': 'control',
    'Streetcar (Stop for)': 'control'
}
dataset['TRAFFCTL'] = dataset['TRAFFCTL'].replace(mapping_TRAFFCTL)

mapping_DRIVACT = {
    'Driving Properly': 'Normal',
    'Lost control': 'Not Normal',
    'Failed to Yield Right of Way': 'Not Normal',
    'Improper Passing': 'Not Normal',
    'Improper Turn': 'Not Normal',
    'Exceeding Speed Limit': 'Not Normal',
    'Disobeyed Traffic Control': 'Not Normal',
    'Following too Close': 'Not Normal',
    'Other': 'Not Normal',
    'Improper Lane Change': 'Not Normal',
    'Wrong Way on One Way Road': 'Not Normal',
    'Speed too Fast For Condition': 'Not Normal',
    'Speed too Slow': 'Not Normal'
}

dataset['DRIVACT'] = dataset['DRIVACT'].replace(mapping_DRIVACT)

mapping_DRIVCOND = {
    'Normal': 'Normal',
  'Ability Impaired, Alcohol Over .08': 'Not Normal',
  'Unknown': 'Not Normal',
  'Inattentive': 'Not Normal',
  'Had Been Drinking': 'Not Normal',
  'Medical or Physical Disability': 'Not Normal',
  'Ability Impaired, Alcohol': 'Not Normal',
  'Fatigue': 'Not Normal',
  'Other': 'Not Normal',
  'Ability Impaired, Drugs': 'Not Normal'
}

dataset['DRIVCOND'] = dataset['DRIVCOND'].replace(mapping_DRIVCOND)

# df_to_csv(dataset, 'modified.csv')

for i in dataset.columns:
    print("------------------------------")
    print(i)    
    print(dataset[i].unique())


dataset_exp_b = expolaration(dataset)

x = dataset
labels = data.ACCLASS
y = np.where((labels == 'Fatal'), 1 , 0)
yy = pd.DataFrame(y)
x_exp = expolaration(x)

array_one = ['TIME']
array_two = ['ROAD_CLASS','DISTRICT','LOCCOORD','TRAFFCTL','VISIBILITY','LIGHT','RDSFCOND',
            'INVAGE','INJURY','DRIVACT','DRIVCOND']
array_three = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH',
              'PASSENGER','SPEEDING','AG_DRIV','REDLIGHT','ALCOHOL','DISABILITY']

num_attribs = array_one
cat_attribs = array_two
where_attribs = array_three

where_transformer = WhereTransformer(columns= array_three)
num_pipeline_gitansh = Pipeline([('scaler',  StandardScaler())])

transformer = ColumnTransformer(transformers = [
        ("num_pipe", num_pipeline_gitansh, num_attribs),
        ("num", MostFrequentImputer(), cat_attribs)],remainder='passthrough'
)

transformer2 = ColumnTransformer([
        ("num_pipe", num_pipeline_gitansh, num_attribs),
        ("one_hot", OneHotEncoder(handle_unknown='ignore'), cat_attribs),
        ("where", where_transformer, where_attribs),

])


r = transformer.fit_transform(x)
r = pd.DataFrame(r, columns = x.columns)
l = transformer2.fit_transform(r)


num_cols = num_attribs
cat_cols = transformer2.named_transformers_['one_hot'].get_feature_names_out(cat_attribs)
where_cols = where_attribs
all_cols = num_cols + list(cat_cols) + where_cols
final = pd.DataFrame(l, columns= all_cols)

smote = SMOTE(random_state=42)
X_final, y_final = smote.fit_resample(final, y)

##### graphical representation of data

data.hist(bins=50, figsize = (20,15))

final.hist(bins=50, figsize = (20,15))


sns.heatmap(data=final, cmap='coolwarm')


df = dataset.copy()

for i in array_three:
    df[i] = np.where((df[i] == 'Yes'),1,0)


for i in array_two:
    df[i], _ = pd.factorize(df[i])
    
    
# for correlation
df = pd.concat([df,yy],axis=1)

df.hist(bins=50, figsize = (20,15))

sns.countplot(x = 0 , data=df)
for i in df.columns:
    sns.countplot(x = i , data=df)
    plt.show()
    
for i in df.columns:
    sns.countplot(x = i , hue = 0, data=df)
    plt.show()

co_relation = df.corr()
relation = co_relation[0].sort_values(ascending = True)
print(relation)
df_to_csv(co_relation, 'corr_matrix.csv')

sns.clustermap(co_relation, cmap='coolwarm', annot=True)

sns.heatmap(data=co_relation,cmap='coolwarm')

from pandas.plotting import scatter_matrix

scatter_matrix(df, figsize=(12, 8))
############################################################

