import urllib
import numpy as np
import pandas as pd

from random import seed
import os

# methods for training, and analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# import classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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

# X_train = DSX_train[['age_cat_binary', 'is_recid', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'charge_degree_binary', 'sex_binary']]
# X_test = DSX_test[['age_cat_binary', 'is_recid', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'charge_degree_binary', 'sex_binary']]
X_train = DSX_train[['is_recid', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'charge_degree_binary']]
X_test = DSX_test[['is_recid', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'charge_degree_binary']]

y_train = DSy_train
y_test = DSy_test

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=33)

# preparing dfs for analysis
df_african_american = pd.DataFrame(None, columns=['two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'knn', 'dtc', 'xgb', 'rafo', 'lreg', 'svm'])
df_caucasian = pd.DataFrame(None, columns=['two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'knn', 'dtc', 'xgb', 'rafo', 'lreg', 'svm'])
df_hispanic = pd.DataFrame(None, columns=['two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'knn', 'dtc', 'xgb', 'rafo', 'lreg', 'svm'])
df_asian = pd.DataFrame(None, columns=['two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'knn', 'dtc', 'xgb', 'rafo', 'lreg', 'svm'])
df_native_american = pd.DataFrame(None, columns=['two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'knn', 'dtc', 'xgb', 'rafo', 'lreg', 'svm'])
df_other = pd.DataFrame(None, columns=['two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'knn', 'dtc', 'xgb', 'rafo', 'lreg', 'svm'])

df_male = pd.DataFrame(None, columns=['two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'knn', 'dtc', 'xgb', 'rafo', 'lreg', 'svm'])
df_female = pd.DataFrame(None, columns=['two_year_recid', 'decile_score', 'is_5_or_more_decile_score', 'gnb', 'knn', 'dtc', 'xgb', 'rafo', 'lreg', 'svm'])

#for the different rates from the classifiers
df_compas_rates = pd.DataFrame(None, columns=['Caucasian', 'African-American', 'Hispanic', 'Asian', 'Native American', 'Other', 'Male', 'Female'],
                        index=['tpr', 'tnr', 'fpr', 'fnr', 'acc'])
df_gnb_rates = pd.DataFrame(None, columns=['Caucasian', 'African-American', 'Hispanic', 'Asian', 'Native American', 'Other', 'Male', 'Female'],
                        index=['tpr', 'tnr', 'fpr', 'fnr', 'acc'])
df_knn_rates = pd.DataFrame(None, columns=['Caucasian', 'African-American', 'Hispanic', 'Asian', 'Native American', 'Other', 'Male', 'Female'],
                        index=['tpr', 'tnr', 'fpr', 'fnr', 'acc'])
df_dtc_rates = pd.DataFrame(None, columns=['Caucasian', 'African-American', 'Hispanic', 'Asian', 'Native American', 'Other', 'Male', 'Female'],
                        index=['tpr', 'tnr', 'fpr', 'fnr', 'acc'])
df_xgb_rates = pd.DataFrame(None, columns=['Caucasian', 'African-American', 'Hispanic', 'Asian', 'Native American', 'Other', 'Male', 'Female'],
                        index=['tpr', 'tnr', 'fpr', 'fnr', 'acc'])
df_rafo_rates = pd.DataFrame(None, columns=['Caucasian', 'African-American', 'Hispanic', 'Asian', 'Native American', 'Other', 'Male', 'Female'],
                        index=['tpr', 'tnr', 'fpr', 'fnr', 'acc'])
df_lreg_rates = pd.DataFrame(None, columns=['Caucasian', 'African-American', 'Hispanic', 'Asian', 'Native American', 'Other', 'Male', 'Female'],
                        index=['tpr', 'tnr', 'fpr', 'fnr', 'acc'])
df_svm_rates = pd.DataFrame(None, columns=['Caucasian', 'African-American', 'Hispanic', 'Asian', 'Native American', 'Other', 'Male', 'Female'],
                        index=['tpr', 'tnr', 'fpr', 'fnr', 'acc'])

#for the overall
result = pd.DataFrame
df_concl = pd.DataFrame(DSX_test, columns=['id', 'sex', 'race', 'two_year_recid', 'decile_score', 'is_5_or_more_decile_score'])
df_cl_comp = pd.DataFrame(None, columns=['compas', 'gnb', 'knn', 'dtc', 'xgb', 'rafo', 'lreg', 'svm'],
                          index=['tpr', 'tnr', 'fpr', 'fnr', 'acc'])


# Classifiers
gnb = GaussianNB()
K = 6
knn = KNeighborsClassifier(n_neighbors = K)
dtc = DecisionTreeClassifier()
#from Arthur
xgb = XGBClassifier(verbosity=0)
rafo = RandomForestClassifier()
lreg = LogisticRegression()
svm = SVC()
classifiers = [gnb, knn, dtc, xgb, rafo, lreg, svm]
classifiersStr = ['gnb', 'knn', 'dtc', 'xgb', 'rafo', 'lreg', 'svm']
classifiersRates = [df_gnb_rates, df_knn_rates, df_dtc_rates, df_xgb_rates, df_rafo_rates, df_lreg_rates, df_svm_rates]
# classifiers = [gnb, knn]
# classifiersStr = ['gnb', 'knn']
# classifiersRates = [df_gnb_rates, df_knn_rates]

# methods
def fitting(classifier):
    classifier.fit(X_train, y_train)

def predicting(classifier, classifierName):
    y_model = classifier.predict(X_test)
    df_cl_comp.loc['acc', classifierName] = format(accuracy_score(y_test, y_model), '.2%')
    conf = confusion_matrix(y_test, y_model)
    df_cl_comp.loc['tnr', classifierName] = format(conf[0, 0] / (conf[0, 0] + conf[0, 1]), '.2%')
    df_cl_comp.loc['tpr', classifierName] = format(conf[1, 1] / (conf[1, 1] + conf[1, 0]), '.2%')
    df_cl_comp.loc['fnr', classifierName] = format(conf[1, 0] / (conf[1, 0] + conf[1, 1]), '.2%')
    df_cl_comp.loc['fpr', classifierName] = format(conf[0, 1] / (conf[0, 1] + conf[0, 0]), '.2%')
    if type(classifier) == GaussianNB:
        df_concl['gnb'] = y_model
    elif type(classifier) == KNeighborsClassifier:
        df_concl['knn'] = y_model
    elif type(classifier) == DecisionTreeClassifier:
        df_concl['dtc'] = y_model
    elif type(classifier) == XGBClassifier:
        df_concl['xgb'] = y_model
    elif type(classifier) == RandomForestClassifier:
        df_concl['rafo'] = y_model
    elif type(classifier) == LogisticRegression:
        df_concl['lreg'] = y_model
    elif type(classifier) == SVC:
        df_concl['svm'] = y_model


def printrate(race, conf, rates, acc):
    rates.loc['tnr', race] = format(conf.loc[0, 0] / (conf.loc[0, 0] + conf.loc[0, 1]), '.2%')
    rates.loc['tpr', race] = format(conf.loc[1, 1] / (conf.loc[1, 1] + conf.loc[1, 0]), '.2%')
    rates.loc['fnr', race] = format(conf.loc[1, 0] / (conf.loc[1, 0] + conf.loc[1, 1]), '.2%')
    rates.loc['fpr', race] = format(conf.loc[0, 1] / (conf.loc[0, 1] + conf.loc[0, 0]), '.2%')
    rates.loc['acc', race] = format(acc, '.2%')

def fairness(classifier, rates):
    #print('\n rates from classifier ', classifier)
    conf = pd.crosstab(df_caucasian['two_year_recid'], df_caucasian[classifier],
                                   rownames=['Actual'], colnames=['Predicted'])
    printrate('Caucasian', conf, rates, accuracy_score(df_caucasian['two_year_recid'].tolist(), df_caucasian[classifier].tolist()))
    conf = pd.crosstab(df_african_american['two_year_recid'], df_african_american[classifier], rownames=['Actual'],
                                   colnames=['Predicted'])
    printrate('African-American', conf, rates, accuracy_score(df_african_american['two_year_recid'].tolist(), df_african_american[classifier].tolist()))
    conf = pd.crosstab(df_hispanic['two_year_recid'], df_hispanic[classifier], rownames=['Actual'],
                       colnames=['Predicted'])
    printrate('Hispanic', conf, rates, accuracy_score(df_hispanic['two_year_recid'].tolist(), df_hispanic[classifier].tolist()))
    conf = pd.crosstab(df_asian['two_year_recid'], df_asian[classifier], rownames=['Actual'],
                       colnames=['Predicted'])
    printrate('Asian', conf, rates, accuracy_score(df_asian['two_year_recid'].tolist(), df_asian[classifier].tolist()))
    conf = pd.crosstab(df_native_american['two_year_recid'], df_native_american[classifier], rownames=['Actual'],
                       colnames=['Predicted'])
    printrate('Native American', conf, rates, accuracy_score(df_native_american['two_year_recid'].tolist(), df_native_american[classifier].tolist()))
    conf = pd.crosstab(df_other['two_year_recid'], df_other[classifier], rownames=['Actual'],
                       colnames=['Predicted'])
    printrate('Other', conf, rates, accuracy_score(df_other['two_year_recid'].tolist(), df_other[classifier].tolist()))
    conf = pd.crosstab(df_male['two_year_recid'], df_male[classifier], rownames=['Actual'], colnames=['Predicted'])
    printrate('Male', conf, rates, accuracy_score(df_male['two_year_recid'].tolist(), df_male[classifier].tolist()))
    conf = pd.crosstab(df_female['two_year_recid'], df_female[classifier], rownames=['Actual'], colnames=['Predicted'])
    printrate('Female', conf, rates, accuracy_score(df_female['two_year_recid'].tolist(), df_female[classifier].tolist()))


################ classifier creation and application and fairness analyzation

#print('\n compas accuracy: ', format(accuracy_score(df_concl.two_year_recid, df_concl.is_5_or_more_decile_score), '.2%'))
df_cl_comp.loc['acc', 'compas'] = format(accuracy_score(df_concl.two_year_recid, df_concl.is_5_or_more_decile_score), '.2%')
conf = confusion_matrix(df_concl.two_year_recid, df_concl.is_5_or_more_decile_score)
df_cl_comp.loc['tnr', 'compas'] = format(conf[0, 0] / (conf[0, 0] + conf[0, 1]), '.2%')
df_cl_comp.loc['tpr', 'compas'] = format(conf[1, 1] / (conf[1, 1] + conf[1, 0]), '.2%')
df_cl_comp.loc['fnr', 'compas'] = format(conf[1, 0] / (conf[1, 0] + conf[1, 1]), '.2%')
df_cl_comp.loc['fpr', 'compas'] = format(conf[0, 1] / (conf[0, 1] + conf[0, 0]), '.2%')


i=0
while i < len(classifiers):
    fitting(classifiers[i])
    predicting(classifiers[i], classifiersStr[i])
    i+=1

print('\nComparison of Classifiers\n', df_cl_comp.to_string())

# creates dfs for subgroups to observe in terms of fairness
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


fairness('is_5_or_more_decile_score', df_compas_rates)

i=0
while i < len(classifiers):
    fairness(classifiersStr[i], classifiersRates[i])
    #print('\n', classifiersStr)
    #print(classifiersRates[i].to_string())
    i+=1

data = {
    'compas_w': df_compas_rates['Caucasian'],
    'compas_b': df_compas_rates['African-American'],
    'gnb_w': df_gnb_rates['Caucasian'],
    'gnb_b': df_gnb_rates['African-American'],
    'knn_w': df_knn_rates['Caucasian'],
    'knn_b': df_knn_rates['African-American'],
    'dtc_w': df_dtc_rates['Caucasian'],
    'dtc_b': df_dtc_rates['African-American'],
    'xgb_w': df_xgb_rates['Caucasian'],
    'xbg_b': df_xgb_rates['African-American'],
    'rafo_w': df_rafo_rates['Caucasian'],
    'rafo_b': df_rafo_rates['African-American'],
    'lreg_w': df_lreg_rates['Caucasian'],
    'lreg_b': df_lreg_rates['African-American'],
    'svm_w': df_svm_rates['Caucasian'],
    'svm_b': df_svm_rates['African-American']
}

result = pd.DataFrame(data, index=['tpr', 'tnr', 'fpr', 'fnr', 'acc'])
print('\n Comparison of False predicted Values in COMPAS and different Classifiers\n', result.to_string())
