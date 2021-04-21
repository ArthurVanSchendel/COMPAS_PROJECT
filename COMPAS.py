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
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

SEED=1234
seed(SEED)
np.random.seed(SEED)

def check_data_file(fname):
    files=os.listdir(".")
    print("Looking for gile '%s' in the current directory...", fname)

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

# add classifiers here :
xgb = XGBClassifier()
Random_forest = RandomForestClassifier()
Logistic_reg = LogisticRegression()
svm_classifier = SVC()
Gauss_NB = GaussianNB()




dataset = pd.read_csv(COMPAS_INPUT_FILE)
dataset = dataset.dropna(subset=["days_b_screening_arrest"]) # dropping missing vals
dataset = dataset[(dataset.days_b_screening_arrest <= 30) & 
(dataset.days_b_screening_arrest >= -30) &
(dataset.is_recid != -1) & (dataset.c_charge_degree != 'O') & (dataset.score_text != 'N/A')]
dataset.reset_index(inplace=True, drop=True) # renumber the rows from 0 again

# turn into a binary classification problem
# create feature is_med_or_high_risk
dataset['is_5_or_more_decile_score']  = (dataset['decile_score']>=5).astype(int)
dataset['is_med_or_high_risk'] = (dataset['score_text']!='Low').astype(int)   # combine medium and high risk
dataset['age_cat_binary'] = (dataset['age']<=35).astype(int)           # below 35 y.o = 1 / above 35 = 0
dataset['sex_binary'] = (dataset['sex']=='Male').astype(int)           # male = 1, female = 0
dataset['charge_degree_binary'] = (dataset['c_charge_degree']=='F').astype(int)   #felony = 1 / misdemeanor = 0

X = dataset[['age_cat_binary', 'is_recid', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'charge_degree_binary']]
#X = dataset[['is_5_or_more_decile_score', 'is_med_or_high_risk', 'age_cat_binary', 'sex_binary']]
#X = dataset[['age', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'is_recid','charge_degree_binary', 'sex_binary']]
y = dataset.two_year_recid

compas_pred = []
acc_compas = 0
for i in range(len(dataset)):
    if dataset.decile_score[i] >= 5:
        compas_pred.append(1)
    else:
        compas_pred.append(0)

tp = 0
fp = 0
fn = 0
tn = 0

tp_afr = 0
fp_afr = 0
fn_afr = 0
tn_afr = 0

tp_asian = 0
fp_asian = 0
fn_asian = 0
tn_asian = 0

tp_hispanic = 0
fp_hispanic = 0
fn_hispanic = 0
tn_hispanic = 0

tp_other = 0
fp_other = 0
fn_other = 0
tn_other = 0

tp_native = 0
fp_native = 0
fn_native = 0
tn_native = 0

tp_caucasian = 0
fp_caucasian = 0
fn_caucasian = 0
tn_caucasian = 0

for j in range(len(dataset)):
    if compas_pred[j] == 1 and dataset.two_year_recid[j] == 1:
        acc_compas+=1
        tp +=1

        if (dataset.race[j] == 'African-American'):
            tp_afr +=1
        elif (dataset.race[j] == 'Other'):
            tp_other+=1
        elif (dataset.race[j] == 'Hispanic'):
            tp_hispanic+=1
        elif (dataset.race[j] == 'Asian'):
            tp_asian+=1
        elif (dataset.race[j] == 'Native American'):
            tp_native+=1
        else:           # caucasian
            tp_caucasian+=1

    elif compas_pred[j] == 1 and dataset.two_year_recid[j] == 0:
        fn += 1
        if (dataset.race[j] == 'African-American'):
            fn_afr +=1
        elif (dataset.race[j] == 'Other'):
            fn_other+=1
        elif (dataset.race[j] == 'Hispanic'):
            fn_hispanic+=1
        elif (dataset.race[j] == 'Asian'):
            fn_asian+=1
        elif (dataset.race[j] == 'Native American'):
            fn_native+=1
        else:           # caucasian
            fn_caucasian+=1
    elif compas_pred[j] == 0 and dataset.two_year_recid[j] == 1:
        fp += 1
        if (dataset.race[j] == 'African-American'):
            fp_afr +=1
        elif (dataset.race[j] == 'Other'):
            fp_other+=1
        elif (dataset.race[j] == 'Hispanic'):
            fp_hispanic+=1
        elif (dataset.race[j] == 'Asian'):
            fp_asian+=1
        elif (dataset.race[j] == 'Native American'):
            fp_native+=1
        else:           # caucasian
            fp_caucasian+=1
    else:
        tn += 1
        acc_compas+=1
        if (dataset.race[j] == 'African-American'):
            tn_afr +=1
        elif (dataset.race[j] == 'Other'):
            tn_other+=1
        elif (dataset.race[j] == 'Hispanic'):
            tn_hispanic+=1
        elif (dataset.race[j] == 'Asian'):
            tn_asian+=1
        elif (dataset.race[j] == 'Native American'):
            tn_native+=1
        else:           # caucasian
            tn_caucasian+=1


acc_compas = (acc_compas/len(dataset))*100
print("\n accuracy of compas classifier = ", acc_compas)
print("\n")
print("\n true positive of compas classifier = ", tp)
print("\n")
print("\n true negative of compas classifier = ", tn)
print("\n")
print("\n fale positive of compas classifier = ", fp)
print("\n")
print("\n false negative of compas classifier = ", fn)
print("\n")

print("\n true positive for african american = ", tp_afr)
print("\n true negative for african american = ", tn_afr)
print("\n false positive for african american =", fp_afr)
print("\n false negative for african amercian = ", fn_afr)

print("\n true positive for asian = ", tp_asian)
print("\n true negative for asian = ", tn_asian)
print("\n false positive for asian =", fp_asian)
print("\n false negative for asian = ", fn_asian)

print("\n true positive for hispanic = ", tp_hispanic)
print("\n true negative for hispanic = ", tn_hispanic)
print("\n false positive for hispanic =", fp_hispanic)
print("\n false negative for hispanic = ", fn_hispanic)

print("\n true positive for caucasian = ", tp_caucasian)
print("\n true negative for caucasian = ", tn_caucasian)
print("\n false positive for caucasian =", fp_caucasian)
print("\n false negative for caucasian = ", fn_caucasian)

print("\n true positive for other = ", tp_other)
print("\n true negative for other = ", tn_other)
print("\n false positive for other =", fp_other)
print("\n false negative for other = ", fn_other)

print("\n true positive for native = ", tp_native)
print("\n true negative for native = ", tn_native)
print("\n false positive for native =", fp_native)
print("\n false negative for native = ", fn_native)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=33)

#train classifiers here
##############################################   TRAINING   ####################################################################
xgb.fit(X_train, y_train)
Random_forest.fit(X_train, y_train)
Logistic_reg.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)
Gauss_NB.fit(X_train, y_train)

###############################################  PREDICTING  ##################################################################

xgb_pred = xgb.predict(X_test)
rf_pred = Random_forest.predict(X_test)
lr_pred = Logistic_reg.predict(X_test)
svm_pred = svm_classifier.predict(X_test)

print("\n len(xgb_pred) = ", len(xgb_pred))
print("\n len(X) = ", len(X))
print("\n len(rf_pred) = ", len(rf_pred))
print("\n len(lr_pred) = ", len(lr_pred))
print("\n len(svm_pred) = ", len(svm_pred))

###############################################  EVALUATING  ##################################################################

print(accuracy_score(y_test, xgb_pred))
print(confusion_matrix(y_test, xgb_pred))

print(accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))

print(accuracy_score(y_test, lr_pred))
print(confusion_matrix(y_test, lr_pred))


print(accuracy_score(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))

acc_xgb_afr = 0
acc_xgb_hispanic = 0
acc_xgb_other = 0
acc_xgb_caucasian = 0
acc_xgb_native = 0
acc_xgb_asian = 0

cnt_xgb_afr = 0
cnt_xgb_hispanic = 0
cnt_xgb_other = 0
cnt_xgb_caucasian = 0
cnt_xgb_native = 0
cnt_xgb_asian = 0

acc_lr_afr = 0
acc_lr_hispanic = 0
acc_lr_other = 0
acc_lr_caucasian = 0
acc_lr_native = 0
acc_lr_asian = 0

cnt_lr_afr = 0
cnt_lr_hispanic = 0
cnt_lr_other = 0
cnt_lr_caucasian = 0
cnt_lr_native = 0
cnt_lr_asian = 0

print("\n y test = ", y_test)
#print("\n xgb_pred = ", xgb_pred)

print("\n y_test[1] = ", y_test.values)
#print("\n xgb_pred[50] = ", xgb_pred[50])
print("\n y_test shape = ", y_test.shape)
print("\n xgb_pred shape = ", xgb_pred.shape)
print("\n len y_test = ", len(y_test))
#print("\n y_test[10][1] = ", y_test[10][1])

for j in range (len(y_test)):
    if (dataset.race[j] == 'Other'):
        cnt_xgb_other+=1
        cnt_lr_other+=1
        if (xgb_pred[j] == y_test.values[j]):
            acc_xgb_other+=1
        if (lr_pred[j] == y_test.values[j]):
            acc_lr_other+=1
    if (dataset.race[j] == 'African-American'):
        cnt_xgb_afr+=1
        cnt_lr_afr+=1
        if (xgb_pred[j] == y_test.values[j]):
            acc_xgb_afr+=1
        if (lr_pred[j] == y_test.values[j]):
            acc_lr_afr+=1
    if (dataset.race[j] == 'Native American'):
        cnt_xgb_native+=1
        cnt_lr_native+=1
        if (xgb_pred[j] == y_test.values[j]):
            acc_xgb_native+=1
        if (lr_pred[j] == y_test.values[j]):
            acc_lr_native+=1
    if (dataset.race[j] == 'Asian'):
        cnt_xgb_asian+=1
        cnt_lr_asian+=1
        if (xgb_pred[j] == y_test.values[j]):
            acc_xgb_asian+=1
        if (lr_pred[j] == y_test.values[j]):
            acc_lr_asian+=1
    if (dataset.race[j] == 'Caucasian'):
        cnt_xgb_caucasian+=1
        cnt_lr_caucasian+=1
        if (xgb_pred[j] == y_test.values[j]):
            acc_xgb_caucasian+=1
        if (lr_pred[j] == y_test.values[j]):
            acc_lr_caucasian+=1
    if (dataset.race[j] == 'Hispanic'):
        cnt_xgb_hispanic+=1
        cnt_lr_hispanic+=1
        if (xgb_pred[j] == y_test.values[j]):
            acc_xgb_hispanic+=1
        if (lr_pred[j] == y_test.values[j]):
            acc_lr_hispanic+=1

acc_xgb_afr = (acc_xgb_afr/cnt_xgb_afr)*100
acc_xgb_hispanic = (acc_xgb_hispanic/cnt_xgb_hispanic)*100
acc_xgb_other = (acc_xgb_other/cnt_xgb_other)*100
acc_xgb_caucasian = (acc_xgb_caucasian/cnt_xgb_caucasian)*100
acc_xgb_native = (acc_xgb_native/cnt_xgb_native)*100
acc_xgb_asian = (acc_xgb_asian/cnt_xgb_asian)*100

acc_lr_afr = (acc_lr_afr/cnt_lr_afr)*100
acc_lr_hispanic = (acc_lr_hispanic/cnt_lr_hispanic)*100
acc_lr_other = (acc_lr_other/cnt_lr_other)*100
acc_lr_caucasian = (acc_lr_caucasian/cnt_xgb_caucasian)*100
acc_lr_native = (acc_lr_native/cnt_xgb_native)*100
acc_lr_asian = (acc_lr_asian/cnt_xgb_asian)*100

print("\n XGB: afr = ", acc_xgb_afr)
print("\n XGB: hispanic = ", acc_xgb_hispanic)
print("\n XGB: other = ", acc_xgb_other)
print("\n XGB: caucasian = ", acc_xgb_caucasian)
print("\n XGB: native = ", acc_xgb_native)
print("\n XGB: asian = ", acc_xgb_asian)

print("\n LR: afr = ", acc_lr_afr)
print("\n LR: hispanic = ", acc_lr_hispanic)
print("\n LR: other = ", acc_lr_other)
print("\n LR: caucasian = ", acc_lr_caucasian)
print("\n LR: native = ", acc_lr_native)
print("\n LR: asian = ", acc_lr_asian)


###############################################   STATISTICS  #################################################################

all_races = []
j = 0
parsed = False
counter=0

cnt_minus25 = 0
cnt_25_45 = 0
cnt_greater_45 = 0

cnt_caucas = 0
cnt_other = 0
cnt_afro_american = 0
cnt_hispanic = 0
cnt_asian = 0
cnt_native_american = 0

cnt_dec_score1 = 0
cnt_dec_score2 = 0
cnt_dec_score3 = 0
cnt_dec_score4 = 0
cnt_dec_score5 = 0
cnt_dec_score6 = 0
cnt_dec_score7 = 0
cnt_dec_score8 = 0
cnt_dec_score9 = 0
cnt_dec_score10 = 0

cnt_low = 0
cnt_medium = 0
cnt_high = 0

avg_compas_score_caucasian_male = 0
avg_compas_score_caucasian_female = 0
cnt_caucas_male = 0
cnt_caucas_female = 0

avg_compas_score_other_male = 0
avg_compas_score_other_female = 0
cnt_other_male = 0
cnt_other_female = 0

avg_compas_score_afro_male = 0
avg_compas_score_afro_female = 0
cnt_afro_male = 0
cnt_afro_female = 0

avg_compas_score_hispanic_male = 0
avg_compas_score_hispanic_female = 0
cnt_hispanic_male = 0
cnt_hispanic_female = 0

avg_compas_score_asian_male = 0
avg_compas_score_asian_female = 0
cnt_asian_male = 0
cnt_asian_female = 0


avg_compas_score_native_male = 0
avg_compas_score_native_female = 0
cnt_native_male = 0
cnt_native_female = 0


cnt_recid = 0
cnt_not_recid = 0

cnt_recid_caucas_male = 0
cnt_recid_caucas_female = 0

cnt_recid_other_male = 0
cnt_recid_other_female = 0

cnt_recid_afro_male = 0
cnt_recid_afro_female = 0

cnt_recid_hispanic_male = 0
cnt_recid_hispanic_female = 0

cnt_recid_asian_male = 0
cnt_recid_asian_female = 0

cnt_recid_native_male = 0
cnt_recid_native_female = 0

compas_accuracy = 0

compas_accuracy_caucas_male = 0
compas_accuracy_caucas_female = 0

compas_accuracy_other_male = 0
compas_accuracy_other_female = 0

compas_accuracy_afro_male = 0
compas_accuracy_afro_female = 0

compas_accuracy_hispanic_male = 0
compas_accuracy_hispanic_female = 0

compas_accuracy_asian_male = 0
compas_accuracy_asian_female = 0

compas_accuracy_native_male = 0
compas_accuracy_native_female = 0

false_positive = 0
false_negative = 0
true_positive = 0
true_negative = 0

for i in range (0, dataset.shape[0]):

    for j in range(0, len(all_races)):
        parsed=False
        if (dataset.race[i] not in all_races):
            counter+=1
    if (counter == len(all_races)):
        parsed = True
        all_races.append(dataset.race[i])

    if (dataset.two_year_recid[i] == 1 & dataset.is_recid[i] == 0):
        false_negative+=1
    if (dataset.two_year_recid[i] == 0 & dataset.is_recid[i] == 1):
        false_positive+=1
    if (dataset.two_year_recid[i] == 1 & dataset.is_recid[i] == 1):
        true_positive+=1
    if (dataset.two_year_recid[i] == 0 & dataset.is_recid[i] == 0):
        true_negative+=1

    if (dataset.race[i] == 'Caucasian'):
        cnt_caucas+=1
        if (dataset.sex[i] == 'Male'):
            cnt_caucas_male+=1
            avg_compas_score_caucasian_male+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_caucas_male+=1

            if (dataset.two_year_recid[i] == dataset.is_recid[i]):
                compas_accuracy_caucas_male+=1

        if (dataset.sex[i] == 'Female'):
            cnt_caucas_female+=1
            avg_compas_score_caucasian_female+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_caucas_female+=1

            if (dataset.two_year_recid[i] == dataset.is_recid[i]):
                compas_accuracy_caucas_female+=1

    if (dataset.race[i] == 'Other'):
        cnt_other+=1
        if (dataset.sex[i] == 'Male'):
            cnt_other_male+=1
            avg_compas_score_other_male+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_other_male+=1
            if (dataset.two_year_recid[i] == dataset.is_recid[i]):
                compas_accuracy_other_male+=1
        if (dataset.sex[i] == 'Female'):
            cnt_other_female+=1
            avg_compas_score_other_female+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_other_female+=1
            if (dataset.two_year_recid[i] == dataset.is_recid[i]):
                compas_accuracy_other_female+=1
    if (dataset.race[i] == 'African-American'):
        cnt_afro_american+=1
        if (dataset.sex[i] == 'Male'):
            cnt_afro_male+=1
            avg_compas_score_afro_male+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_afro_male+=1
            if (dataset.two_year_recid[i] == dataset.is_recid[i]):
                compas_accuracy_afro_male+=1
        if (dataset.sex[i] == 'Female'):
            cnt_afro_female+=1
            avg_compas_score_afro_female+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_afro_female+=1
            if (dataset.two_year_recid[i] == dataset.is_recid[i]):
                compas_accuracy_afro_female+=1
    if (dataset.race[i] == 'Hispanic'):
        cnt_hispanic+=1
        if (dataset.sex[i] == 'Male'):
            cnt_hispanic_male+=1
            avg_compas_score_hispanic_male+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_hispanic_male+=1
            if (dataset.two_year_recid[i] == dataset.is_recid[i]):
                compas_accuracy_hispanic_male+=1
        if (dataset.sex[i] == 'Female'):
            cnt_hispanic_female+=1
            avg_compas_score_hispanic_female+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_hispanic_female+=1
            if (dataset.two_year_recid[i] == dataset.is_recid[i]):
                compas_accuracy_hispanic_female+=1
    if (dataset.race[i] == 'Asian'):
        cnt_asian+=1
        if (dataset.sex[i] == 'Male'):
            cnt_asian_male+=1
            avg_compas_score_asian_male+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_asian_male+=1
            if (dataset.two_year_recid[i] == dataset.is_recid[i]):
                compas_accuracy_asian_male+=1
        if (dataset.sex[i] == 'Female'):
            cnt_asian_female+=1
            avg_compas_score_asian_female+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_asian_female+=1
            if (dataset.two_year_recid[i] == dataset.is_recid[i]):
                compas_accuracy_asian_female+=1
    if (dataset.race[i] == 'Native American'):
        cnt_native_american+=1
        if (dataset.sex[i] == 'Male'):
            cnt_native_male+=1
            avg_compas_score_native_male+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_native_male+=1
            if (dataset.two_year_recid[i] == dataset.is_recid[i]):
                compas_accuracy_native_male+=1
        if (dataset.sex[i] == 'Female'):
            cnt_native_female+=1
            avg_compas_score_native_female+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_native_female+=1
            if (dataset.two_year_recid[i] == dataset.is_recid[i]):
                compas_accuracy_native_female+=1

    if (dataset.two_year_recid[i] == 1):
        cnt_recid+=1
    else:
        cnt_not_recid+=1

    if (dataset.two_year_recid[i] == dataset.is_recid[i]):
        compas_accuracy+=1




    if(dataset.age_cat[i] == '25 - 45'):
        cnt_25_45+=1
    if (dataset.age_cat[i] == 'Greater than 45'):
        cnt_greater_45+=1
    if (dataset.age_cat[i] == 'Less than 25'):
        #less than 25
        cnt_minus25+=1

    if (dataset.decile_score[i] == 1):
        cnt_dec_score1+=1
    if (dataset.decile_score[i] == 2):
        cnt_dec_score2+=1
    if (dataset.decile_score[i] == 3):
        cnt_dec_score3+=1
    if (dataset.decile_score[i] == 4):
        cnt_dec_score4+=1
    if (dataset.decile_score[i] == 5):
        cnt_dec_score5+=1
    if (dataset.decile_score[i] == 6):
        cnt_dec_score6+=1
    if (dataset.decile_score[i] == 7):
        cnt_dec_score7+=1
    if (dataset.decile_score[i] == 8):
        cnt_dec_score8+=1
    if (dataset.decile_score[i] == 9):
        cnt_dec_score9+=1
    if (dataset.decile_score[i] == 10):
        cnt_dec_score10+=1

    if (dataset.score_text[i] == 'Low'):
        cnt_low+=1
    if (dataset.score_text[i] == 'Medium'):
        cnt_medium+=1
    if (dataset.score_text[i] == 'High'):
        cnt_high+=1



    counter = 0


print("\n number of people below 25 years old: ", cnt_minus25)
print("\n number of people in between 25 and 45: ", cnt_25_45)
print("\n number of people above 45 years old: ", cnt_greater_45)

print("\n total number of people = ", cnt_25_45+cnt_greater_45+cnt_minus25)

print("\n caucasian count = ", cnt_caucas)
print("\n other count = ", cnt_other)
print("\n afro american count = ", cnt_afro_american)
print("\n hispanic count = ", cnt_hispanic)
print("\n asian count = ", cnt_asian)
print("\n native american count = ", cnt_native_american)

print("\n nb of people with decile score = 1 : ", cnt_dec_score1)
print("\n nb of people with decile score = 2 : ", cnt_dec_score2)
print("\n nb of people with decile score = 3 : ", cnt_dec_score3)
print("\n nb of people with decile score = 4 : ", cnt_dec_score4)
print("\n nb of people with decile score = 5 : ", cnt_dec_score5)
print("\n nb of people with decile score = 6 : ", cnt_dec_score6)
print("\n nb of people with decile score = 7 : ", cnt_dec_score7)
print("\n nb of people with decile score = 8 : ", cnt_dec_score8)
print("\n nb of people with decile score = 9 : ", cnt_dec_score9)
print("\n nb of people with decile score = 10 : ", cnt_dec_score10)


print("\n nb of people with low risk : ", cnt_low)
print("\n nb of people with medium risk : ", cnt_medium)
print("\n nb of people with high risk : ", cnt_high)

print("\n caucasian avg decile score (M / F) = ", avg_compas_score_caucasian_male/cnt_caucas_male, avg_compas_score_caucasian_female/cnt_caucas_female)
print("\n other avg decile score (M / F) = ", avg_compas_score_other_male/cnt_other_male, avg_compas_score_other_female/cnt_other_female)
print("\n afro american avg decile score (M / F) = ", avg_compas_score_afro_male/cnt_afro_male, avg_compas_score_afro_female/cnt_afro_female)
print("\n hispanic avg decile score ( M / F ) = ", avg_compas_score_hispanic_male/cnt_hispanic_male, avg_compas_score_hispanic_female/cnt_hispanic_female)
print("\n asian avg decile score (M / F ) = ", avg_compas_score_asian_male/cnt_asian_male, avg_compas_score_asian_female/cnt_asian_female)
print("\n native american avg decile score (M / F ) = ", avg_compas_score_native_male/cnt_native_male, avg_compas_score_native_female/cnt_native_female)

print("\n nb of people recidivist within 2 years = ", cnt_recid) 
print("\n nb of people NOT recidivist within 2 years = ", cnt_not_recid)   

print("\n recidivism caucasian ( M / F ) =", (cnt_recid_caucas_male/cnt_caucas_male)*100, (cnt_recid_caucas_female/cnt_caucas_female)*100)
print("\n recidivism other ( M / F ) = ", (cnt_recid_other_male/cnt_other_male)*100, (cnt_recid_other_female/cnt_other_female)*100)
print("\n recidivism afro american ( M / F ) = ", (cnt_recid_afro_male/cnt_afro_male)*100, (cnt_recid_afro_female/cnt_afro_female)*100)
print("\n recidivism hispanic ( M / F ) = ", (cnt_recid_hispanic_male/cnt_hispanic_male)*100, (cnt_recid_hispanic_female/cnt_hispanic_female)*100)
print("\n recidivism asian ( M / F ) = ", (cnt_recid_asian_male/cnt_asian_male)*100, (cnt_recid_asian_female/cnt_asian_female)*100)
print("\n recidivism native american ( M / F ) =", (cnt_recid_native_male/cnt_native_male)*100, (cnt_recid_native_female/cnt_native_female)*100)

print("\n accuracy of COMPAS dataset = ", (compas_accuracy/dataset.shape[0])*100)
print("\n accuracy of COMPAS dataset for caucasians = ", (compas_accuracy_caucas_male/cnt_caucas_male)*100, (compas_accuracy_caucas_female/cnt_caucas_female)*100)
print("\n accuracy of COMPAS dataset for other = ", (compas_accuracy_other_male/cnt_other_male)*100, (compas_accuracy_other_female/cnt_other_female)*100)
print("\n accuracy of COMPAS dataset for afro = ", (compas_accuracy_afro_male/cnt_afro_male)*100, (compas_accuracy_afro_female/cnt_afro_female)*100)
print("\n accuracy of COMPAS dataset for hispanic = ", (compas_accuracy_hispanic_male/cnt_hispanic_male)*100, (compas_accuracy_hispanic_female/cnt_hispanic_female)*100)
print("\n accuracy of COMPAS dataset for asian = ", (compas_accuracy_asian_male/cnt_asian_male)*100, (compas_accuracy_asian_female/cnt_asian_female)*100)
print("\n accuracy of COMPAS dataset for native = ", (compas_accuracy_native_male/cnt_native_male)*100, (compas_accuracy_native_female/cnt_native_female)*100)
print("\n false negative = ", false_negative)
print("\n false positive = ", false_positive)
print("\n true negative = ", true_negative)
print("\n true positive = ", true_positive)
print("\n all races = ", all_races)
