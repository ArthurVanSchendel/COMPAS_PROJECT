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


# outputting

def printresult(testarray, modelarray):
    c=0
    for i,j in zip(testarray, modelarray):
        if c<20:
            print(i, ' - ', j)
        else:
            break
        c += 1



X = dataset[['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'is_recid', 'is_violent_recid']]
y = dataset.two_year_recid


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=33)

#print(y_test.keys)


# Gausian Naive Bayes Classifier -> 97%
print('GausianNB')
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_gnb = gnb.predict(X_test)
score_gnb = accuracy_score(y_test, y_gnb)
#print(score_gnb)
#print(confusion_matrix(y_test, y_gnb))
#printresult(y_test, y_gnb)
total_gnb = gnb.predict(X)
print(confusion_matrix(y, total_gnb))
#printresult(y, total_gnb)


#kNN - K=3 -> 93% slightly better than 2 (91%) what doesn't make sense.
print('kNN')
K = 3
knn = KNeighborsClassifier(n_neighbors = K)
knn.fit(X_train, y_train)
y_knn = knn.predict(X_test)
print(accuracy_score(y_test, y_knn))
#print(confusion_matrix(y_test, y_knn))
#printresult(y_test, y_knn)
total_knn = knn.predict(X)
print(confusion_matrix(y, total_knn))
#printresult(y, total_knn)

#randomforest
#decisiontree
#mlp


################### look at fairness

df_african_american = pd.DataFrame(None, columns=['id', 'race', 'two_year_recid', 'decile_score', 'gnb', 'kNN'])
df_caucasian = pd.DataFrame(None, columns=['id', 'race', 'two_year_recid', 'decile_score', 'gnb', 'kNN'])

#dataset_cleaned = dataset[['id', 'race', 'two_year_recid', 'decile_score']]

df_concl = pd.DataFrame(dataset, columns=['id', 'race', 'two_year_recid', 'decile_score'])

df_concl['gnb'] = total_gnb
df_concl['kNN'] = total_knn

#df_concl['compas'] =


for index, row in df_concl.iterrows():
    if row['race'] == 'Caucasian':
        df_caucasian = df_caucasian.append(row, ignore_index=True)
    elif row['race'] == 'African-American':
        df_african_american = df_african_american.append(row, ignore_index=True)



#compas
print('compas: white then black')
confusion_matrix = pd.crosstab(df_caucasian['two_year_recid'], df_caucasian['compas'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

confusion_matrix = pd.crosstab(df_african_american['two_year_recid'], df_african_american['compas'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

#gnb
print('gnb: white then black')
confusion_matrix = pd.crosstab(df_caucasian['two_year_recid'], df_caucasian['gnb'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

confusion_matrix = pd.crosstab(df_african_american['two_year_recid'], df_african_american['gnb'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

#knn
print('knn: white then black')
confusion_matrix = pd.crosstab(df_caucasian['two_year_recid'], df_caucasian['kNN'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

confusion_matrix = pd.crosstab(df_african_american['two_year_recid'], df_african_american['kNN'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)


#
# #print(df_caucasian)
# groundtruth = df_caucasian['two_year_recid']
# gnb_vals = df_caucasian['gnb']
# print(gnb_vals)
# print(confusion_matrix(groundtruth, gnb_vals))