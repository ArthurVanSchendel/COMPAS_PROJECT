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

X = dataset[['is_recid', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'charge_degree_binary', 'is_violent_recid', 'juv_other_count']] #unbiased dataset
#X = dataset[['is_recid', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'charge_degree_binary']] #biased dataset
y = dataset.two_year_recid

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y, random_state=33)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False, random_state=33)

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

fn_xgb_caucas = 0
fp_xgb_caucas = 0
fn_rf_caucas = 0
fp_rf_caucas = 0
fn_lr_caucas = 0
fp_lr_caucas = 0
fn_svm_caucas = 0
fp_svm_caucas = 0

fn_xgb_hispanic = 0
fp_xgb_hispanic = 0
fn_rf_hispanic = 0
fp_rf_hispanic = 0
fn_lr_hispanic = 0
fp_lr_hispanic = 0
fn_svm_hispanic = 0
fp_svm_hispanic = 0

fn_xgb_asian = 0
fp_xgb_asian = 0
fn_rf_asian = 0
fp_rf_asian = 0
fn_lr_asian = 0
fp_lr_asian = 0
fn_svm_asian = 0
fp_svm_asian = 0

fn_xgb_afr = 0
fp_xgb_afr = 0
fn_rf_afr = 0
fp_rf_afr = 0
fn_lr_afr = 0
fp_lr_afr = 0
fn_svm_afr = 0
fp_svm_afr = 0

fn_xgb_native = 0
fp_xgb_native = 0
fn_rf_native = 0
fp_rf_native = 0
fn_lr_native = 0
fp_lr_native = 0
fn_svm_native = 0
fp_svm_native = 0

fn_xgb_other = 0
fp_xgb_other = 0
fn_rf_other = 0
fp_rf_other = 0
fn_lr_other = 0
fp_lr_other = 0
fn_svm_other = 0
fp_svm_other = 0

tn_xgb_caucas = 0
tp_xgb_caucas = 0
tn_rf_caucas = 0
tp_rf_caucas = 0
tn_lr_caucas = 0
tp_lr_caucas = 0
tn_svm_caucas = 0
tp_svm_caucas = 0

tn_xgb_hispanic = 0
tp_xgb_hispanic = 0
tn_rf_hispanic = 0
tp_rf_hispanic = 0
tn_lr_hispanic = 0
tp_lr_hispanic = 0
tn_svm_hispanic = 0
tp_svm_hispanic = 0

tn_xgb_asian = 0
tp_xgb_asian = 0
tn_rf_asian = 0
tp_rf_asian = 0
tn_lr_asian = 0
tp_lr_asian = 0
tn_svm_asian = 0
tp_svm_asian = 0

tn_xgb_afr = 0
tp_xgb_afr = 0
tn_rf_afr = 0
tp_rf_afr = 0
tn_lr_afr = 0
tp_lr_afr = 0
tn_svm_afr = 0
tp_svm_afr = 0

tn_xgb_native = 0
tp_xgb_native = 0
tn_rf_native = 0
tp_rf_native = 0
tn_lr_native = 0
tp_lr_native = 0
tn_svm_native = 0
tp_svm_native = 0

tn_xgb_other = 0
tp_xgb_other = 0
tn_rf_other = 0
tp_rf_other = 0
tn_lr_other = 0
tp_lr_other = 0
tn_svm_other = 0
tp_svm_other = 0

cnt_afr = 0
cnt_caucas = 0
cnt_hispanic = 0
cnt_other = 0
cnt_native = 0
cnt_asian = 0

for i in range(len(y_test)):
    if (dataset.race[i] == 'Asian'):
        cnt_asian+=1
    if (dataset.race[i] == 'Caucasian'):
        cnt_caucas+=1
    if (dataset.race[i] == 'African-American'):
        cnt_afr+=1
    if (dataset.race[i] == 'Other'):
        cnt_other+=1
    if(dataset.race[i] == 'Hispanic'):
        cnt_hispanic+=1
    if (dataset.race[i] == 'Native American'):
        cnt_native+=1

    if(dataset.two_year_recid[i+4629] == 1 and xgb_pred[i] == 1):
        #true positive
        if (dataset.race[i]=='African-American'):
            tp_xgb_afr+=1
        if (dataset.race[i]=='Asian'):
            tp_xgb_asian+=1
        if (dataset.race[i]=='Caucasian'):
            tp_xgb_caucas+=1
        if (dataset.race[i]=='Other'):
            tp_xgb_other+=1
        if (dataset.race[i]=='Hispanic'):
            tp_xgb_hispanic+=1
        if (dataset.race[i]=='Native American'):
            tp_xgb_native+=1
    if(dataset.two_year_recid[i+4629] == 0 and xgb_pred[i] == 0):
        #true negative
        if (dataset.race[i]=='African-American'):
            tn_xgb_afr+=1
        if (dataset.race[i]=='Asian'):
            tn_xgb_asian+=1
        if (dataset.race[i]=='Caucasian'):
            tn_xgb_caucas+=1
        if (dataset.race[i]=='Other'):
            tn_xgb_other+=1
        if (dataset.race[i]=='Hispanic'):
            tn_xgb_hispanic+=1
        if (dataset.race[i]=='Native American'):
            tn_xgb_native+=1
        
    if(dataset.two_year_recid[i+4629] == 1 and lr_pred[i] == 1):
        #true positive
        if (dataset.race[i]=='African-American'):
            tp_lr_afr+=1
        if (dataset.race[i]=='Asian'):
            tp_lr_asian+=1
        if (dataset.race[i]=='Caucasian'):
            tp_lr_caucas+=1
        if (dataset.race[i]=='Other'):
            tp_lr_other+=1
        if (dataset.race[i]=='Hispanic'):
            tp_lr_hispanic+=1
        if (dataset.race[i]=='Native American'):
            tp_lr_native+=1

    if(dataset.two_year_recid[i+4629] == 0 and lr_pred[i] == 0):
        #true negative
        if (dataset.race[i]=='African-American'):
            tn_lr_afr+=1
        if (dataset.race[i]=='Asian'):
            tn_lr_asian+=1
        if (dataset.race[i]=='Caucasian'):
            tn_lr_caucas+=1
        if (dataset.race[i]=='Other'):
            tn_lr_other+=1
        if (dataset.race[i]=='Hispanic'):
            tn_lr_hispanic+=1
        if (dataset.race[i]=='Native American'):
            tn_lr_native+=1

    if(dataset.two_year_recid[i+4629] == 1 and rf_pred[i] == 1):
        #true positive
        if (dataset.race[i]=='African-American'):
            tp_rf_afr+=1
        if (dataset.race[i]=='Asian'):
            tp_rf_asian+=1
        if (dataset.race[i]=='Caucasian'):
            tp_rf_caucas+=1
        if (dataset.race[i]=='Other'):
            tp_rf_other+=1
        if (dataset.race[i]=='Hispanic'):
            tp_rf_hispanic+=1
        if (dataset.race[i]=='Native American'):
            tp_rf_native+=1

    if(dataset.two_year_recid[i+4629] == 0 and rf_pred[i] == 0):
        #true negative
        if (dataset.race[i]=='African-American'):
            tn_rf_afr+=1
        if (dataset.race[i]=='Asian'):
            tn_rf_asian+=1
        if (dataset.race[i]=='Caucasian'):
            tn_rf_caucas+=1
        if (dataset.race[i]=='Other'):
            tn_rf_other+=1
        if (dataset.race[i]=='Hispanic'):
            tn_rf_hispanic+=1
        if (dataset.race[i]=='Native American'):
            tn_rf_native+=1

    if(dataset.two_year_recid[i+4629] == 1 and svm_pred[i] == 1):
        #true positive
        if (dataset.race[i]=='African-American'):
            tp_svm_afr+=1
        if (dataset.race[i]=='Asian'):
            tp_svm_asian+=1
        if (dataset.race[i]=='Caucasian'):
            tp_svm_caucas+=1
        if (dataset.race[i]=='Other'):
            tp_svm_other+=1
        if (dataset.race[i]=='Hispanic'):
            tp_svm_hispanic+=1
        if (dataset.race[i]=='Native American'):
            tp_svm_native+=1

    if(dataset.two_year_recid[i+4629] == 0 and svm_pred[i] == 0):
        #true negative
        if (dataset.race[i]=='African-American'):
            tn_svm_afr+=1
        if (dataset.race[i]=='Asian'):
            tn_svm_asian+=1
        if (dataset.race[i]=='Caucasian'):
            tn_svm_caucas+=1
        if (dataset.race[i]=='Other'):
            tn_svm_other+=1
        if (dataset.race[i]=='Hispanic'):
            tn_svm_hispanic+=1
        if (dataset.race[i]=='Native American'):
            tn_svm_native+=1

    if (dataset.two_year_recid[i+4629] == 1 and xgb_pred[i] == 0):
        #false negative
        if (dataset.race[i]=='African-American'):
            fn_xgb_afr+=1
        if (dataset.race[i]=='Asian'):
            fn_xgb_asian+=1
        if (dataset.race[i]=='Caucasian'):
            fn_xgb_caucas+=1
        if (dataset.race[i]=='Other'):
            fn_xgb_other+=1
        if (dataset.race[i]=='Hispanic'):
            fn_xgb_hispanic+=1
        if (dataset.race[i]=='Native American'):
            fn_xgb_native+=1
    if (dataset.two_year_recid[i+4629] == 0 and xgb_pred[i] == 1):
        #false positive
        if (dataset.race[i]=='African-American'):
            fp_xgb_afr+=1
        if (dataset.race[i]=='Asian'):
            fp_xgb_asian+=1
        if (dataset.race[i]=='Caucasian'):
            fp_xgb_caucas+=1
        if (dataset.race[i]=='Other'):
            fp_xgb_other+=1
        if (dataset.race[i]=='Hispanic'):
            fp_xgb_hispanic+=1
        if (dataset.race[i]=='Native American'):
            fp_xgb_native+=1

    if (dataset.two_year_recid[i+4629] == 1 and rf_pred[i] == 0):
        #false negative
        if (dataset.race[i]=='African-American'):
            fn_rf_afr+=1
        if (dataset.race[i]=='Asian'):
            fn_rf_asian+=1
        if (dataset.race[i]=='Caucasian'):
            fn_rf_caucas+=1
        if (dataset.race[i]=='Other'):
            fn_rf_other+=1
        if (dataset.race[i]=='Hispanic'):
            fn_rf_hispanic+=1
        if (dataset.race[i]=='Native American'):
            fn_rf_native+=1
    if (dataset.two_year_recid[i+4629] == 0 and rf_pred[i] == 1):
        #false positive
        if (dataset.race[i]=='African-American'):
            fp_rf_afr+=1
        if (dataset.race[i]=='Asian'):
            fp_rf_asian+=1
        if (dataset.race[i]=='Caucasian'):
            fp_rf_caucas+=1
        if (dataset.race[i]=='Other'):
            fp_rf_other+=1
        if (dataset.race[i]=='Hispanic'):
            fp_rf_hispanic+=1
        if (dataset.race[i]=='Native American'):
            fp_rf_native+=1
    
    if (dataset.two_year_recid[i+4629] == 1 and lr_pred[i] == 0):
        #false negative
        if (dataset.race[i]=='African-American'):
            fn_lr_afr+=1
        if (dataset.race[i]=='Asian'):
            fn_lr_asian+=1
        if (dataset.race[i]=='Caucasian'):
            fn_lr_caucas+=1
        if (dataset.race[i]=='Other'):
            fn_lr_other+=1
        if (dataset.race[i]=='Hispanic'):
            fn_lr_hispanic+=1
        if (dataset.race[i]=='Native American'):
            fn_lr_native+=1
    if (dataset.two_year_recid[i+4629] == 0 and lr_pred[i] == 1):
        #false positive
        if (dataset.race[i]=='African-American'):
            fp_lr_afr+=1
        if (dataset.race[i]=='Asian'):
            fp_lr_asian+=1
        if (dataset.race[i]=='Caucasian'):
            fp_lr_caucas+=1
        if (dataset.race[i]=='Other'):
            fp_lr_other+=1
        if (dataset.race[i]=='Hispanic'):
            fp_lr_hispanic+=1
        if (dataset.race[i]=='Native American'):
            fp_lr_native+=1

    if (dataset.two_year_recid[i+4629] == 1 and svm_pred[i] == 0):
        #false negative
        if (dataset.race[i]=='African-American'):
            fn_svm_afr+=1
        if (dataset.race[i]=='Asian'):
            fn_svm_asian+=1
        if (dataset.race[i]=='Caucasian'):
            fn_svm_caucas+=1
        if (dataset.race[i]=='Other'):
            fn_svm_other+=1
        if (dataset.race[i]=='Hispanic'):
            fn_svm_hispanic+=1
        if (dataset.race[i]=='Native American'):
            fn_svm_native+=1
    if (dataset.two_year_recid[i+4629] == 0 and svm_pred[i] == 1):
        #false positive
        if (dataset.race[i]=='African-American'):
            fp_svm_afr+=1
        if (dataset.race[i]=='Asian'):
            fp_svm_asian+=1
        if (dataset.race[i]=='Caucasian'):
            fp_svm_caucas+=1
        if (dataset.race[i]=='Other'):
            fp_svm_other+=1
        if (dataset.race[i]=='Hispanic'):
            fp_svm_hispanic+=1
        if (dataset.race[i]=='Native American'):
            fp_svm_native+=1

#########  XGB #########
print("\n XGB: False Negative Rate for African Americans = ", (fn_xgb_afr/(fn_xgb_afr+tp_xgb_afr))*100)
print("\n XGB: False Negative Rate for Asians = ", (fn_xgb_asian/(fn_xgb_asian+tp_xgb_asian))*100)
print("\n XGB: False Negative Rate for Caucasians = ", (fn_xgb_caucas/(fn_xgb_caucas+tp_xgb_caucas))*100)
print("\n XGB: False Negative Rate for Others = ", (fn_xgb_other/(fn_xgb_other+tp_xgb_other))*100)
print("\n XGB: False Negative Rate for Hispanic = ", (fn_xgb_hispanic/(fn_xgb_hispanic+tp_xgb_hispanic))*100)
print("\n XGB: False Negative Rate for Native Americans = ", (fn_xgb_native/(fn_xgb_native+tp_xgb_native))*100)

print("\n Log. Reg. : False Negative Rate for African Americans = ", (fn_lr_afr/(fn_lr_afr+tp_lr_afr))*100)
print("\n Log. Reg. : False Negative Rate for Asians = ", (fn_lr_asian/(fn_lr_asian+tp_lr_asian))*100)
print("\n Log. Reg. : False Negative Rate for Caucasians = ", (fn_lr_caucas/(fn_lr_caucas+tp_lr_caucas))*100)
print("\n Log. Reg. : False Negative Rate for Others = ", (fn_lr_other/(fn_lr_other+tp_lr_other))*100)
print("\n Log. Reg. : False Negative Rate for Hispanic = ", (fn_lr_hispanic/(fn_lr_hispanic+tp_lr_hispanic))*100)
print("\n Log. Reg. : False Negative Rate for Native Americans = ", (fn_lr_native/(fn_lr_native+tp_lr_native))*100)

print("\n Rand. For.: False Negative Rate for African Americans = ", (fn_rf_afr/(fn_rf_afr+tp_rf_afr))*100)
print("\n Rand. For.: False Negative Rate for Asians = ", (fn_rf_asian/(fn_rf_asian+tp_rf_asian))*100)
print("\n Rand. For.: False Negative Rate for Caucasians = ", (fn_rf_caucas/(fn_rf_caucas+tp_rf_caucas))*100)
print("\n Rand. For.: False Negative Rate for Others = ", (fn_rf_other/(fn_rf_other+tp_rf_other))*100)
print("\n Rand. For.: False Negative Rate for Hispanic = ", (fn_rf_hispanic/(fn_rf_hispanic+tp_rf_hispanic))*100)
print("\n Rand. For.: False Negative Rate for Native Americans = ", (fn_rf_native/(fn_rf_native+tp_rf_native))*100)

print("\n SVM: False Negative Rate for African Americans = ", (fn_svm_afr/(fn_svm_afr+tp_svm_afr))*100)
print("\n SVM: False Negative Rate for Asians = ", (fn_svm_asian/(fn_svm_asian+tp_svm_asian))*100)
print("\n SVM: False Negative Rate for Caucasians = ", (fn_svm_caucas/(fn_svm_caucas+tp_svm_caucas))*100)
print("\n SVM: False Negative Rate for Others = ", (fn_svm_other/(fn_svm_other+tp_svm_other))*100)
print("\n SVM: False Negative Rate for Hispanic = ", (fn_svm_hispanic/(fn_svm_hispanic+tp_svm_hispanic))*100)
print("\n SVM: False Negative Rate for Native Americans = ", (fn_svm_native/(fn_svm_native+tp_svm_native))*100)

print("\n XGB: False Positive Rate for African Americans = ", (fp_xgb_afr/(fp_xgb_afr+tn_xgb_afr))*100)
print("\n XGB: False Positive Rate for Asians = ", (fp_xgb_asian/(fp_xgb_asian+tn_xgb_asian))*100)
print("\n XGB: False Positive Rate for Caucasians = ", (fp_xgb_caucas/(fp_xgb_caucas+tn_xgb_caucas))*100)
print("\n XGB: False Positive Rate for Others = ", (fp_xgb_other/(fp_xgb_other+tn_xgb_other))*100)
print("\n XGB: False Positive Rate for Hispanic = ", (fp_xgb_hispanic/(fp_xgb_hispanic+tn_xgb_hispanic))*100)
print("\n XGB: False Positive Rate for Native Americans = ", (fp_xgb_native/(fp_xgb_native+tn_xgb_native))*100)

print("\n Log. Reg. : False Positive Rate for African Americans = ", (fp_lr_afr/(fp_lr_afr+tn_lr_afr))*100)
print("\n Log. Reg. : False Positive Rate for Asians = ", (fp_lr_asian/(fp_lr_asian+tn_lr_asian))*100)
print("\n Log. Reg. : False Positive Rate for Caucasians = ", (fp_lr_caucas/(fp_lr_caucas+tn_lr_caucas))*100)
print("\n Log. Reg. : False Positive Rate for Others = ", (fp_lr_other/(fp_lr_other+tn_lr_other))*100)
print("\n Log. Reg. : False Positive Rate for Hispanic = ", (fp_lr_hispanic/(fp_lr_hispanic+tn_lr_hispanic))*100)
print("\n Log. Reg. : False Positive Rate for Native Americans = ", (fp_lr_native/(fp_lr_native+tn_lr_native))*100)

print("\n SVM: False Positive Rate for African Americans = ", (fp_svm_afr/(fp_svm_afr+tn_svm_afr))*100)
print("\n SVM: False Positive Rate for Asians = ", (fp_svm_asian/(fp_svm_asian+tn_svm_asian))*100)
print("\n SVM: False Positive Rate for Caucasians = ", (fp_svm_caucas/(fp_svm_caucas+tn_svm_caucas))*100)
print("\n SVM: False Positive Rate for Others = ", (fp_svm_other/(fp_svm_other+tn_svm_other))*100)
print("\n SVM: False Positive Rate for Hispanic = ", (fp_svm_hispanic/(fp_svm_hispanic+tn_svm_hispanic))*100)
print("\n SVM: False Positive Rate for Native Americans = ", (fp_svm_native/(fp_svm_native+tn_svm_native))*100)

print("\n Rand. For.: False Positive Rate for African Americans = ", (fp_rf_afr/(fp_rf_afr+tn_rf_afr))*100)
print("\n Rand. For.: False Positive Rate for Asians = ", (fp_rf_asian/(fp_rf_asian+tn_rf_asian))*100)
print("\n Rand. For.: False Positive Rate for Caucasians = ", (fp_rf_caucas/(fp_rf_caucas+tn_rf_caucas))*100)
print("\n Rand. For.: False Positive Rate for Others = ", (fp_rf_other/(fp_rf_other+tn_rf_other))*100)
print("\n Rand. For.: False Positive Rate for Hispanic = ", (fp_rf_hispanic/(fp_rf_hispanic+tn_rf_hispanic))*100)
print("\n Rand. For.: False Positive Rate for Native Americans = ", (fp_rf_native/(fp_rf_native+tn_rf_native))*100)






fp_xgb_afr
fp_xgb_asian
fp_xgb_caucas
fp_xgb_other
fp_xgb_hispanic
fp_xgb_native

print("\n ", )

print("\n len(X_train) = ", len(X_train))
print("\n len(X_test) = ", len(X_test))
print("\n len(dataset) = ", len(dataset))
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
