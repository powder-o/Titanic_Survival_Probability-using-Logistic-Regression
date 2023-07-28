# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 04:27:00 2023

@author: dhruv
"""

# TITANIC SURVIVAL USING KAGGLE DATA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore')

df_train = pd.read_csv("C:/Users/dhruv/.spyder-py3/Machine Learning sreeni/Logistic Reg/titanic_kaggle/train.csv")
df_test = pd.read_csv("C:/Users/dhruv/.spyder-py3/Machine Learning sreeni/Logistic Reg/titanic_kaggle/test.csv")
print(df_train.head())

#plt.scatter(df_train.Pclass, df_train.Survived)
#plt.scatter(df_train.Fare, df_train.Survived)
#plt.scatter(df_train.Embarked, df_train.Survived)

# Looking at Null values
print('total null values:',df_train.isnull().sum())

# Exploring all features

print('percentage of null values in Age: ',(df_train.Age.isnull().sum()/df_train.shape[0])*100) #AGE
print('percentage of null values in Cabin: ',(df_train.Cabin.isnull().sum()/df_train.shape[0])*100) #Cabin
print('percentage of null values in Embarked: ',(df_train.Embarked.isnull().sum()/df_train.shape[0])*100) #Embarked

print('mean of age: ',df_train.Age.mean(skipna=True))
print('median of age: ',df_train.Age.median(skipna=True))
df_train['Age'].hist(bins=15, density=True, stacked=True)

sns.countplot(x='Embarked', data = df_train)
print('Embarked data distribution:',df_train.Embarked.value_counts())

# Doing what we need to do after Exploring features

train_data = df_train.copy()

train_data['Age'].fillna(train_data['Age'].median(skipna=True), inplace=True)   #replacing nan with age median

train_data['Embarked'].fillna(train_data['Embarked'].value_counts().idxmax(), inplace=True)  #replacing nan with most occured value

train_data.drop('Cabin', axis=1, inplace=True)   #dropping cabin

train_data.drop('PassengerId', axis=1, inplace=True)   #dropping passengerId

train_data.drop('Name', axis=1, inplace=True)   #dropping name

train_data.drop('Ticket', axis=1, inplace=True)

train_data.head()
print(train_data.isnull().sum())

# tuning other features
train_data['Travelling_alone'] = np.where(train_data['SibSp']+train_data['Parch']>0, 0, 1)   #parch and Sibsp are related to if you are travelling alone or not
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)


train_data.Sex[train_data.Sex == 'male'] = 1
train_data.Sex[train_data.Sex == 'female'] = 0

train_data.Embarked[train_data.Embarked == 'S'] = 3
train_data.Embarked[train_data.Embarked == 'C'] = 2
train_data.Embarked[train_data.Embarked == 'Q'] = 1

train_data.Age[train_data.Age <= 16] = 1    # Minor
train_data.Age[train_data.Age > 16] = 0     # Major

Y = train_data.Survived
X = train_data.drop(labels=['Survived'], axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(x_train, y_train)

predicition = model.predict(x_test)
from sklearn import metrics
print('Accuracy % = ', metrics.accuracy_score(y_test, predicition)*100)












