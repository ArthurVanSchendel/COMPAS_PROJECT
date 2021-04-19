import urllib
import os, sys
import numpy as np
import pandas as pd

from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle
import os


from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

SEED=1234
seed(SEED)
np.random.seed(SEED)

def check_data_file(fname):
    files=os.listdir(".")
    print("Looking for file '%s' in the current directory...", fname)

    if fname not in files:
        print("'%s' not found! Downloading from GitHub...", fname)
        addr="https://Raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        #response=urllib.request.urlopen(addr)
        response = urllib.request.urlopen(addr)
        data=response.read()
        fileOut = open(fname, "wb")
        fileOut.write(data)
        fileOut.close()
        print("'%s' download and saved locally..", fname)
    else:
        print("File not found in current directory..")

COMPAS_INPUT_FILE = "compas-scores-two-years.csv"
#check_data_file(COMPAS_INPUT_FILE)



dataset = pd.read_csv(COMPAS_INPUT_FILE)
dataset = dataset.dropna(subset=["days_b_screening_arrest"]) # dropping missing vals
dataset = dataset[(dataset.days_b_screening_arrest <= 30) &
(dataset.days_b_screening_arrest >= -30) &
(dataset.is_recid != -1) & (dataset.c_charge_degree != 'O') & (dataset.score_text != 'N/A')]
dataset.reset_index(inplace=True, drop=True) # renumber the rows from 0 again

X = dataset[['id', 'age', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'is_recid']]
y = dataset.two_year_recid


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y, random_state=33)

#Gausian Bayes Classifier -> 97%
model = GaussianNB()
model.fit(X_train, y_train)
y_model = model.predict(X_test)
accuracy_score(y_test, y_model)
print(accuracy_score(y_test, y_model))
print(confusion_matrix(y_test, y_model))

#kNN - doesn't work at all
K = 2
model2 = KNeighborsClassifier(n_neighbors = K)
model2.fit(X_train, y_train)
y_model2 = model2.predict(X_test)
accuracy_score(y_test, y_model2)
print(accuracy_score(y_test, y_model2))
#compute the confusion matrix
print(confusion_matrix(y_test, y_model2))

