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

DSX_train, DSX_test, DSy_train, DSy_test = train_test_split(dataset, dataset['two_year_recid'], test_size=0.25, shuffle=True, random_state=33)

X_train = DSX_train[['age_cat_binary', 'is_recid', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'charge_degree_binary']]
X_test = DSX_test[['age_cat_binary', 'is_recid', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'charge_degree_binary']]
y_train = DSy_train
y_test = DSy_test

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=33)
# cnt_caucasian = sum(dataset.race == 'Caucasian')

# preparing dfs
df_african_american = pd.DataFrame(None, columns=['id', 'race', 'two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'knn', 'dtc'])
df_caucasian = pd.DataFrame(None, columns=['id', 'race', 'two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'knn', 'dtc'])
df_hispanic = pd.DataFrame(None, columns=['id', 'race', 'two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'knn', 'dtc'])
df_asian = pd.DataFrame(None, columns=['id', 'race', 'two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'knn', 'dtc'])
df_native_american = pd.DataFrame(None, columns=['id', 'race', 'two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'knn', 'dtc'])
df_other = pd.DataFrame(None, columns=['id', 'race', 'two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'knn', 'dtc'])

df_male = pd.DataFrame(None, columns=['id', 'race', 'two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'knn', 'dtc'])
df_female = pd.DataFrame(None, columns=['id', 'race', 'two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'knn', 'dtc'])

df_concl = pd.DataFrame(DSX_test, columns=['id', 'sex', 'race', 'two_year_recid', 'decile_score', 'is_5_or_more_decile_score'])

# Classifiers
gnb = GaussianNB()
K = 6
knn = KNeighborsClassifier(n_neighbors = K)
dtc = DecisionTreeClassifier()
classifiers = [gnb, knn, dtc]
classifiersStr = ['gnb', 'knn', 'dtc']

# methods
def fitting(classifier):
    classifier.fit(X_train, y_train)

def predicting(classifier):
    y_model = classifier.predict(X_test)
    print('\n accuracy score ', type(classifier), ': ', format(accuracy_score(y_test, y_model), '.2%'))
    conf = confusion_matrix(y_test, y_model)
    print('tnr = ', format(conf[0, 0] / (conf[0, 0] + conf[0, 1]), '.2%'), '\n tpr = ',
          format(conf[1, 1] / (conf[1, 1] + conf[1, 0]), '.2%'), '\n fnr = ',
          format(conf[1, 0] / (conf[1, 0] + conf[1, 1]), '.2%'), '\n fpr = ',
          format(conf[0, 1] / (conf[0, 1] + conf[0, 0]), '.2%'))
    #print('fpr: ', format(conf[0, 1]/(conf[0,1]+conf[0,0]),'.2%'), '\nfn: ', format(conf[1, 0]/(conf[1,1]+conf[0,0]),'.2%'))
    if type(classifier) == GaussianNB:
        df_concl['gnb'] = y_model
    elif type(classifier) == KNeighborsClassifier:
        df_concl['knn'] = y_model
    elif type(classifier) == DecisionTreeClassifier:
        df_concl['dtc'] = y_model

def printrate(name, conf):
    print('\n ', name, '\n', 'tnr = ', format(conf.loc[0, 0] / (conf.loc[0, 0] + conf.loc[0, 1]),'.2%'), '\n tpr = ',
          format(conf.loc[1, 1] / (conf.loc[1, 1] + conf.loc[1, 0]),'.2%'), '\n fnr = ',
          format(conf.loc[1, 0] / (conf.loc[1, 0] + conf.loc[1, 1]),'.2%'), '\n fpr = ',
          format(conf.loc[0, 1] / (conf.loc[0, 1] + conf.loc[0, 0]),'.2%'))

def fairness(classifier):
    print('\n rates from classifier ', classifier)
    conf = pd.crosstab(df_caucasian['two_year_recid'], df_caucasian[classifier],
                                   rownames=['Actual'], colnames=['Predicted'])
    printrate('Caucasian: ', conf)
    conf = pd.crosstab(df_african_american['two_year_recid'], df_african_american[classifier], rownames=['Actual'],
                                   colnames=['Predicted'])
    printrate('African-American', conf)
    conf = pd.crosstab(df_hispanic['two_year_recid'], df_hispanic[classifier], rownames=['Actual'],
                       colnames=['Predicted'])
    printrate('Hispanic', conf)
    conf = pd.crosstab(df_asian['two_year_recid'], df_asian[classifier], rownames=['Actual'],
                       colnames=['Predicted'])
    printrate('Asian', conf)
    conf = pd.crosstab(df_native_american['two_year_recid'], df_native_american[classifier], rownames=['Actual'],
                       colnames=['Predicted'])
    printrate('Native-American', conf)
    conf = pd.crosstab(df_other['two_year_recid'], df_other[classifier], rownames=['Actual'],
                       colnames=['Predicted'])
    printrate('Other', conf)
    conf = pd.crosstab(df_male['two_year_recid'], df_male[classifier], rownames=['Actual'], colnames=['Predicted'])
    printrate('Male', conf)
    conf = pd.crosstab(df_female['two_year_recid'], df_female[classifier], rownames=['Actual'], colnames=['Predicted'])
    printrate('Female', conf)

################ classifier creation and application and fairness analyzation

print('\n compas accuracy: ', format(accuracy_score(df_concl.two_year_recid, df_concl.is_5_or_more_decile_score), '.2%'))

for i in classifiers:
    fitting(i)
    predicting(i)

# creates dfs for subgroups to observe
    for index, row in df_concl.iterrows():
        if row['race'] == 'Caucasian':
            df_caucasian = df_caucasian.append(row, ignore_index=True)
        elif row['race'] == 'African-American':
            df_african_american = df_african_american.append(row, ignore_index=True)
        elif row['race'] == 'Asian':
            df_asian = df_asian.append(row, ignore_index=True)
        elif row['race'] == 'Hispanic':
            df_hispanic = df_hispanic.append(row, ignore_index=True)
        elif row['race'] == 'Native American':
            df_native_american = df_native_american.append(row, ignore_index=True)
        elif row['race'] == 'Other':
            df_other = df_other.append(row, ignore_index=True)
        if row['sex'] == 'Male':
            df_male = df_male.append(row, ignore_index=True)
        elif row['sex'] == 'Female':
            df_female = df_female.append(row, ignore_index=True)


fairness('is_5_or_more_decile_score')

for i in classifiersStr:
    fairness(i)
