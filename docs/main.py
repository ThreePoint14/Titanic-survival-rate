import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import sklearn as sk
import tensorflow as tf

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

train = pd.read_csv(r'/Users/pariszhang/Desktop/Data/titanic/train.csv')
train2 = pd.read_csv(r'/Users/pariszhang/Desktop/Data/titanic/gender_submission.csv')
test = pd.read_csv(r'/Users/pariszhang/Desktop/Data/titanic/test.csv')

char_attribs = train.get(['Sex'])

dftrain = train.drop(labels=['Name', 'Embarked', 'Cabin', 'Ticket', 'Sex', 'Survived'] , axis = 1)

labels = train['Survived'].copy()
data = pd.DataFrame(train, columns = train.columns, index = train.index)
trainData = data.drop(labels=['Name', 'Embarked', 'Cabin', 'Ticket', 'Survived'] , axis = 1)

num_pipeline = Pipeline([ ('imputer', SimpleImputer(strategy  = 'median')),
                          ('scaler', StandardScaler()) ])

full_pipeline = ColumnTransformer([ ('num', num_pipeline, list(dftrain)),
                                    ('cat', OneHotEncoder(), list(char_attribs)) ])

readyData = full_pipeline.fit_transform(trainData)

someData = trainData.iloc[:10]
someLabels = labels.iloc[:10]

someDataPrep = full_pipeline.fit_transform(someData)

linReg = LinearRegression()
linReg.fit(readyData, labels)

print(linReg.predict(someDataPrep))

