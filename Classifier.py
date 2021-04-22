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
from sklearn.tree import DecisionTreeClassifier

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


# turn into a binary classification problem
# create feature is_med_or_high_risk
dataset['age_cat_binary'] = (dataset['age']<=35).astype(int)           # below 35 y.o = 1 / above 35 = 0
dataset['sex_binary'] = (dataset['sex']=='Male').astype(int)           # male = 1, female = 0
dataset['charge_degree_binary'] = (dataset['c_charge_degree']=='F').astype(int)   #felony = 1 / misdemeanor = 0
dataset['is_5_or_more_decile_score']  = (dataset['decile_score']>5).astype(int)

DSX_train, DSX_test, DSy_train, DSy_test = train_test_split(dataset, dataset, test_size=0.3, shuffle=True, random_state=33)


#X = dataset[['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'is_recid', 'is_violent_recid']]
X = dataset[['age_cat_binary', 'is_recid', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'charge_degree_binary']]
#X = dataset[['is_5_or_more_decile_score', 'is_med_or_high_risk', 'age_cat_binary', 'sex_binary']]
#X = dataset[['age', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'is_recid','charge_degree_binary', 'sex_binary']]
y = dataset.two_year_recid

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=33)


# cnt_caucasian = sum(dataset.race == 'Caucasian')
# print(cnt_caucasian)
# cnt_african_american = sum(dataset.race == 'African-American')
# print(cnt_african_american)
# cnt_asian = sum(dataset.race == 'Asian')
# cnt_other = sum(dataset.race == 'Other')
# cnt_hispanic = sum(dataset.race == 'Hispanic')
# cnt_native_american = sum(dataset.race == 'Native-American')

# preparing dfs
df_african_american = pd.DataFrame(None, columns=['id', 'race', 'two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'kNN', 'dtc'])
df_caucasian = pd.DataFrame(None, columns=['id', 'race', 'two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'kNN', 'dtc'])
df_male = pd.DataFrame(None, columns=['id', 'race', 'two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'kNN', 'dtc'])
df_female = pd.DataFrame(None, columns=['id', 'race', 'two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'kNN', 'dtc'])

df_concl = pd.DataFrame(DSX_test, columns=['id', 'sex', 'race', 'two_year_recid', 'decile_score', 'is_5_or_more_decile_score'])
# df_concl['gnb'] = total_gnb
# df_concl['kNN'] = total_knn
# df_concl['dtc'] = total_dtc

# Classifiers
gnb = GaussianNB()
K = 3
knn = KNeighborsClassifier(n_neighbors = K)
dtc = DecisionTreeClassifier()
classifiers = [gnb, knn, dtc]

# methods
def fitting(classifier):
    classifier.fit(X_train, y_train)

def predicting(classifier):
    y_model = classifier.predict(X_test)
    print('accuracy score ', type(classifier), ': ', accuracy_score(y_test, y_model))
    conf = confusion_matrix(y_test, y_model)
    print(conf)
    print('fp: ', conf[0, 1], '\nfn: ', conf[1, 0])
    print('fp: ', (conf[0, 1]/(conf[0,1]+conf[0,0]))*100, '\nfn: ', (conf[1, 0]/(conf[1,1]+conf[0,0]))*100)
    if type(classifier) == GaussianNB:
        df_concl['gnb'] = y_model
    elif type(classifier) == KNeighborsClassifier:
        df_concl['kNN'] = y_model
    elif type(classifier) == DecisionTreeClassifier:
        df_concl['dtc'] = y_model

for i in classifiers:
    fitting(i)
    predicting(i)

################### look at fairness

for index, row in df_concl.iterrows():
    if row['race'] == 'Caucasian':
        df_caucasian = df_caucasian.append(row, ignore_index=True)
    elif row['race'] == 'African-American':
        df_african_american = df_african_american.append(row, ignore_index=True)
    if row['sex'] == 'Male':
        df_male = df_male.append(row, ignore_index=True)
    elif row['sex'] == 'Female':
        df_female = df_female.append(row, ignore_index=True)

print(df_concl)
print(df_caucasian)
print(df_african_american)
print(df_male)
print(df_female)


print('\n compas accuracy: ', accuracy_score(df_concl.two_year_recid, df_concl.is_5_or_more_decile_score))

# for index, row in df_african_american.iterrows():
#     if row['decile_score'] > 5:
#         row['compas'] = int(1)
#     elif row['decile_score'] <= 5:
#         row['compas'] = int(0)
#
# for index, row in df_caucasian.iterrows():
#     if row['decile_score'] > 5:
#         row['compas'] = int(1)
#     elif row['decile_score'] <= 5:
#         row['compas'] = int(0)
#
# for index, row in df_male.iterrows():
#     if row['decile_score'] > 5:
#         row['compas'] = int(1)
#     elif row['decile_score'] <= 5:
#         row['compas'] = int(0)
#
# for index, row in df_female.iterrows():
#     if row['decile_score'] > 5:
#         row['compas'] = int(1)
#     elif row['decile_score'] <= 5:
#         row['compas'] = int(0)

#compas
print('\n', 'compas: white then black, men then women')
confusion_matrix = pd.crosstab(df_caucasian['two_year_recid'], df_caucasian['is_5_or_more_decile_score'], rownames=['Actual'], colnames=['Predicted'])
print('\n', confusion_matrix)
confusion_matrix = pd.crosstab(df_african_american['two_year_recid'], df_african_american['is_5_or_more_decile_score'], rownames=['Actual'], colnames=['Predicted'])
print('\n', confusion_matrix)
print('\n', pd.crosstab(df_male['two_year_recid'], df_male['is_5_or_more_decile_score'], rownames=['Actual'], colnames=['Predicted']))
print('\n', pd.crosstab(df_female['two_year_recid'], df_female['is_5_or_more_decile_score'], rownames=['Actual'], colnames=['Predicted']))

#gnb
print('\n', 'gnb: white then black, men then women')
confusion_matrix = pd.crosstab(df_caucasian['two_year_recid'], df_caucasian['gnb'], rownames=['Actual'], colnames=['Predicted'])
print('\n', confusion_matrix)
confusion_matrix = pd.crosstab(df_african_american['two_year_recid'], df_african_american['gnb'], rownames=['Actual'], colnames=['Predicted'])
print('\n', confusion_matrix)
print('\n', pd.crosstab(df_male['two_year_recid'], df_male['gnb'], rownames=['Actual'], colnames=['Predicted']))
print('\n', pd.crosstab(df_female['two_year_recid'], df_female['gnb'], rownames=['Actual'], colnames=['Predicted']))


#knn
print('\n', 'knn: white then black, men then women')
confusion_matrix = pd.crosstab(df_caucasian['two_year_recid'], df_caucasian['kNN'], rownames=['Actual'], colnames=['Predicted'])
print('\n', confusion_matrix)
confusion_matrix = pd.crosstab(df_african_american['two_year_recid'], df_african_american['kNN'], rownames=['Actual'], colnames=['Predicted'])
print('\n', confusion_matrix)
print('\n', pd.crosstab(df_male['two_year_recid'], df_male['kNN'], rownames=['Actual'], colnames=['Predicted']))
print('\n', pd.crosstab(df_female['two_year_recid'], df_female['kNN'], rownames=['Actual'], colnames=['Predicted']))

#dtc
print('\n', 'dtc: white then black, men then women')
confusion_matrix = pd.crosstab(df_caucasian['two_year_recid'], df_caucasian['dtc'], rownames=['Actual'], colnames=['Predicted'])
print('\n', confusion_matrix)
confusion_matrix = pd.crosstab(df_african_american['two_year_recid'], df_african_american['dtc'], rownames=['Actual'], colnames=['Predicted'])
print('\n', confusion_matrix)
print('\n', pd.crosstab(df_male['two_year_recid'], df_male['dtc'], rownames=['Actual'], colnames=['Predicted']))
print('\n', pd.crosstab(df_female['two_year_recid'], df_female['dtc'], rownames=['Actual'], colnames=['Predicted']))