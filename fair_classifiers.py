import urllib
import os, sys
import numpy as np
import pandas as pd

from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle
import os

import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


COMPAS_INPUT_FILE = "compas-scores-two-years.csv"
dataset = pd.read_csv(COMPAS_INPUT_FILE)
dataset = dataset.dropna(subset=["days_b_screening_arrest"]) # dropping missing vals
dataset = dataset[(dataset.days_b_screening_arrest <= 30) & 
(dataset.days_b_screening_arrest >= -30) &
(dataset.is_recid != -1) & (dataset.c_charge_degree != 'O') & (dataset.score_text != 'N/A')]
dataset.reset_index(inplace=True, drop=True) # renumber the rows from 0 again


# we will be trying 4 different classifiers on the COMPAS dataset
xgb = XGBClassifier()
Random_forest = RandomForestClassifier()
Logistic_reg = LogisticRegression()
svm_classifier = SVC()

dataset['is_5_or_more_decile_score']  = (dataset['decile_score']>=5).astype(int)
dataset['is_med_or_high_risk'] = (dataset['score_text']!='Low').astype(int)   # combine medium and high risk
dataset['age_binary'] = (dataset['age']<=35).astype(int)           # below 35 y.o = 1 / above 35 = 0
dataset['sex_binary'] = (dataset['sex']=='Male').astype(int)           # male = 1, female = 0
dataset['charge_degree_binary'] = (dataset['c_charge_degree']=='F').astype(int)   #felony = 1 / misdemeanor = 0

X = dataset[['is_recid', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'charge_degree_binary']] #unbiased dataset
#X = dataset[['is_recid', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'charge_degree_binary']] #biased dataset
y = dataset.two_year_recid

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y, random_state=33)

#train classifiers here
##############################################   TRAINING   ####################################################################

xgb.fit(X_train, y_train)
Random_forest.fit(X_train, y_train)
Logistic_reg.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)

###############################################  PREDICTING  ###################################################################

xgb_pred = xgb.predict(X_test)
rf_pred = Random_forest.predict(X_test)
lr_pred = Logistic_reg.predict(X_test)
svm_pred = svm_classifier.predict(X_test)

## compute fp and fn for each classifier: per race and per gender
#for i in range(len(y_test)):
print("\n len(X_train) = ", len(X_train))
print("\n len(X_test) = ", len(X_test))
print("\n len(dataset) = ", len(dataset))
print("\n len(X_train) + len(X_test) = ", len(X_train) + len(X_test))
print("\n dataset[4630] = ", dataset.priors_count[4629])
#print("\n X_test[0] = ", X_test.priors_count[0])

print("\n this is X_test = ")
print(X_test)


print(accuracy_score(y_test, xgb_pred))
print(confusion_matrix(y_test, xgb_pred))

print(accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))

print(accuracy_score(y_test, lr_pred))
print(confusion_matrix(y_test, lr_pred))

print(accuracy_score(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))
