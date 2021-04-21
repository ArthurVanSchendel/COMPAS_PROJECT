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



dataset = pd.read_csv(COMPAS_INPUT_FILE)
dataset = dataset.dropna(subset=["days_b_screening_arrest"]) # dropping missing vals
dataset = dataset[(dataset.days_b_screening_arrest <= 30) & 
(dataset.days_b_screening_arrest >= -30) &
(dataset.is_recid != -1) & (dataset.c_charge_degree != 'O') & (dataset.score_text != 'N/A')]
dataset.reset_index(inplace=True, drop=True) # renumber the rows from 0 again

y = dataset.two_year_recid


X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.25, shuffle=True, stratify=y, random_state=33)

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

# for the distribution of decile scores per race
cnt_afr_am_dec_s1 = 0
cnt_afr_am_dec_s2 = 0
cnt_afr_am_dec_s3 = 0
cnt_afr_am_dec_s4 = 0
cnt_afr_am_dec_s5 = 0
cnt_afr_am_dec_s6 = 0
cnt_afr_am_dec_s7 = 0
cnt_afr_am_dec_s8 = 0
cnt_afr_am_dec_s9 = 0
cnt_afr_am_dec_s10 = 0

cnt_caucasian_dec_s1 = 0
cnt_caucasian_dec_s2 = 0
cnt_caucasian_dec_s3 = 0
cnt_caucasian_dec_s4 = 0
cnt_caucasian_dec_s5 = 0
cnt_caucasian_dec_s6 = 0
cnt_caucasian_dec_s7 = 0
cnt_caucasian_dec_s8 = 0
cnt_caucasian_dec_s9 = 0
cnt_caucasian_dec_s10 = 0

cnt_other_dec_s1 = 0
cnt_other_dec_s2 = 0
cnt_other_dec_s3 = 0
cnt_other_dec_s4 = 0
cnt_other_dec_s5 = 0
cnt_other_dec_s6 = 0
cnt_other_dec_s7 = 0
cnt_other_dec_s8 = 0
cnt_other_dec_s9 = 0
cnt_other_dec_s10 = 0

cnt_hispanic_dec_s1 = 0
cnt_hispanic_dec_s2 = 0
cnt_hispanic_dec_s3 = 0
cnt_hispanic_dec_s4 = 0
cnt_hispanic_dec_s5 = 0
cnt_hispanic_dec_s6 = 0
cnt_hispanic_dec_s7 = 0
cnt_hispanic_dec_s8 = 0
cnt_hispanic_dec_s9 = 0
cnt_hispanic_dec_s10 = 0

cnt_asian_dec_s1 = 0
cnt_asian_dec_s2 = 0
cnt_asian_dec_s3 = 0
cnt_asian_dec_s4 = 0
cnt_asian_dec_s5 = 0
cnt_asian_dec_s6 = 0
cnt_asian_dec_s7 = 0
cnt_asian_dec_s8 = 0
cnt_asian_dec_s9 = 0
cnt_asian_dec_s10 = 0

cnt_native_dec_s1 = 0
cnt_native_dec_s2 = 0
cnt_native_dec_s3 = 0
cnt_native_dec_s4 = 0
cnt_native_dec_s5 = 0
cnt_native_dec_s6 = 0
cnt_native_dec_s7 = 0
cnt_native_dec_s8 = 0
cnt_native_dec_s9 = 0
cnt_native_dec_s10 = 0

cnt_male_dec_s1 = 0
cnt_male_dec_s2 = 0
cnt_male_dec_s3 = 0
cnt_male_dec_s4 = 0
cnt_male_dec_s5 = 0
cnt_male_dec_s6 = 0
cnt_male_dec_s7 = 0
cnt_male_dec_s8 = 0
cnt_male_dec_s9 = 0
cnt_male_dec_s10 = 0

cnt_female_dec_s1 = 0
cnt_female_dec_s2 = 0
cnt_female_dec_s3 = 0
cnt_female_dec_s4 = 0
cnt_female_dec_s5 = 0
cnt_female_dec_s6 = 0
cnt_female_dec_s7 = 0
cnt_female_dec_s8 = 0
cnt_female_dec_s9 = 0
cnt_female_dec_s10 = 0


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

for i in range (0, dataset.shape[0]):

    for j in range(0, len(all_races)):
        parsed=False
        if (dataset.race[i] not in all_races):
            counter+=1
    if (counter == len(all_races)):
        parsed = True
        all_races.append(dataset.race[i])

    if (dataset.race[i] == 'Caucasian'):
        cnt_caucas+=1
        if (dataset.sex[i] == 'Male'):
            cnt_caucas_male+=1
            avg_compas_score_caucasian_male+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_caucas_male+=1

        if (dataset.sex[i] == 'Female'):
            cnt_caucas_female+=1
            avg_compas_score_caucasian_female+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_caucas_female+=1

    if (dataset.race[i] == 'Other'):
        cnt_other+=1
        if (dataset.sex[i] == 'Male'):
            cnt_other_male+=1
            avg_compas_score_other_male+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_other_male+=1
        if (dataset.sex[i] == 'Female'):
            cnt_other_female+=1
            avg_compas_score_other_female+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_other_female+=1
    if (dataset.race[i] == 'African-American'):
        cnt_afro_american+=1
        if (dataset.sex[i] == 'Male'):
            cnt_afro_male+=1
            avg_compas_score_afro_male+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_afro_male+=1
        if (dataset.sex[i] == 'Female'):
            cnt_afro_female+=1
            avg_compas_score_afro_female+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_afro_female+=1
    if (dataset.race[i] == 'Hispanic'):
        cnt_hispanic+=1
        if (dataset.sex[i] == 'Male'):
            cnt_hispanic_male+=1
            avg_compas_score_hispanic_male+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_hispanic_male+=1
        if (dataset.sex[i] == 'Female'):
            cnt_hispanic_female+=1
            avg_compas_score_hispanic_female+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_hispanic_female+=1
    if (dataset.race[i] == 'Asian'):
        cnt_asian+=1
        if (dataset.sex[i] == 'Male'):
            cnt_asian_male+=1
            avg_compas_score_asian_male+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_asian_male+=1
        if (dataset.sex[i] == 'Female'):
            cnt_asian_female+=1
            avg_compas_score_asian_female+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_asian_female+=1
    if (dataset.race[i] == 'Native American'):
        cnt_native_american+=1
        if (dataset.sex[i] == 'Male'):
            cnt_native_male+=1
            avg_compas_score_native_male+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_native_male+=1
        if (dataset.sex[i] == 'Female'):
            cnt_native_female+=1
            avg_compas_score_native_female+=dataset.decile_score[i]
            if (dataset.two_year_recid[i] == 1):
                cnt_recid_native_female+=1

    if (dataset.two_year_recid[i] == 1):
        cnt_recid+=1
    else:
        cnt_not_recid+=1




    if(dataset.age_cat[i] == '25 - 45'):
        cnt_25_45+=1
    if (dataset.age_cat[i] == 'Greater than 45'):
        cnt_greater_45+=1
    if (dataset.age_cat[i] == 'Less than 25'):
        #less than 25
        cnt_minus25+=1

    if (dataset.decile_score[i] == 1):
        cnt_dec_score1 += 1
        if (dataset.race[i] == 'African-American'):
            cnt_afr_am_dec_s1 += 1
        if (dataset.race[i] == 'Caucasian'):
            cnt_caucasian_dec_s1 += 1
        if (dataset.race[i] == 'Other'):
            cnt_other_dec_s1 += 1
        if (dataset.race[i] == 'Native American'):
            cnt_native_dec_s1 += 1
        if (dataset.race[i] == 'Hispanic'):
            cnt_hispanic_dec_s1 += 1
        if (dataset.race[i] == 'Asian'):
            cnt_asian_dec_s1 += 1
    if (dataset.decile_score[i] == 2):
        cnt_dec_score2 += 1
        if (dataset.race[i] == 'African-American'):
            cnt_afr_am_dec_s2 += 1
        if (dataset.race[i] == 'Caucasian'):
            cnt_caucasian_dec_s2 += 1
        if (dataset.race[i] == 'Other'):
            cnt_other_dec_s2 += 1
        if (dataset.race[i] == 'Native American'):
            cnt_native_dec_s2 += 1
        if (dataset.race[i] == 'Hispanic'):
            cnt_hispanic_dec_s2 += 1
        if (dataset.race[i] == 'Asian'):
            cnt_asian_dec_s2 += 1
    if (dataset.decile_score[i] == 3):
        cnt_dec_score3 += 1
        if (dataset.race[i] == 'African-American'):
            cnt_afr_am_dec_s3 += 1
        if (dataset.race[i] == 'Caucasian'):
            cnt_caucasian_dec_s3 += 1
        if (dataset.race[i] == 'Other'):
            cnt_other_dec_s3 += 1
        if (dataset.race[i] == 'Native American'):
            cnt_native_dec_s3 += 1
        if (dataset.race[i] == 'Hispanic'):
            cnt_hispanic_dec_s3 += 1
        if (dataset.race[i] == 'Asian'):
            cnt_asian_dec_s3 += 1
    if (dataset.decile_score[i] == 4):
        cnt_dec_score4 += 1
        if (dataset.race[i] == 'African-American'):
            cnt_afr_am_dec_s4 += 1
        if (dataset.race[i] == 'Caucasian'):
            cnt_caucasian_dec_s4 += 1
        if (dataset.race[i] == 'Other'):
            cnt_other_dec_s4 += 1
        if (dataset.race[i] == 'Native American'):
            cnt_native_dec_s4 += 1
        if (dataset.race[i] == 'Hispanic'):
            cnt_hispanic_dec_s4 += 1
        if (dataset.race[i] == 'Asian'):
            cnt_asian_dec_s4 += 1
    if (dataset.decile_score[i] == 5):
        cnt_dec_score5 += 1
        if (dataset.race[i] == 'African-American'):
            cnt_afr_am_dec_s5 += 1
        if (dataset.race[i] == 'Caucasian'):
            cnt_caucasian_dec_s5 += 1
        if (dataset.race[i] == 'Other'):
            cnt_other_dec_s5 += 1
        if (dataset.race[i] == 'Native American'):
            cnt_native_dec_s5 += 1
        if (dataset.race[i] == 'Hispanic'):
            cnt_hispanic_dec_s5 += 1
        if (dataset.race[i] == 'Asian'):
            cnt_asian_dec_s5 += 1
    if (dataset.decile_score[i] == 6):
        cnt_dec_score6 += 1
        if (dataset.race[i] == 'African-American'):
            cnt_afr_am_dec_s6 += 1
        if (dataset.race[i] == 'Caucasian'):
            cnt_caucasian_dec_s6 += 1
        if (dataset.race[i] == 'Other'):
            cnt_other_dec_s6 += 1
        if (dataset.race[i] == 'Native American'):
            cnt_native_dec_s6 += 1
        if (dataset.race[i] == 'Hispanic'):
            cnt_hispanic_dec_s6 += 1
        if (dataset.race[i] == 'Asian'):
            cnt_asian_dec_s6 += 1
    if (dataset.decile_score[i] == 7):
        cnt_dec_score7 += 1
        if (dataset.race[i] == 'African-American'):
            cnt_afr_am_dec_s7 += 1
        if (dataset.race[i] == 'Caucasian'):
            cnt_caucasian_dec_s7 += 1
        if (dataset.race[i] == 'Other'):
            cnt_other_dec_s7 += 1
        if (dataset.race[i] == 'Native American'):
            cnt_native_dec_s7 += 1
        if (dataset.race[i] == 'Hispanic'):
            cnt_hispanic_dec_s7 += 1
        if (dataset.race[i] == 'Asian'):
            cnt_asian_dec_s7 += 1
    if (dataset.decile_score[i] == 8):
        cnt_dec_score8 += 1
        if (dataset.race[i] == 'African-American'):
            cnt_afr_am_dec_s8 += 1
        if (dataset.race[i] == 'Caucasian'):
            cnt_caucasian_dec_s8 += 1
        if (dataset.race[i] == 'Other'):
            cnt_other_dec_s8 += 1
        if (dataset.race[i] == 'Native American'):
            cnt_native_dec_s8 += 1
        if (dataset.race[i] == 'Hispanic'):
            cnt_hispanic_dec_s8 += 1
        if (dataset.race[i] == 'Asian'):
            cnt_asian_dec_s8 += 1
    if (dataset.decile_score[i] == 9):
        cnt_dec_score9 += 1
        if (dataset.race[i] == 'African-American'):
            cnt_afr_am_dec_s9 += 1
        if (dataset.race[i] == 'Caucasian'):
            cnt_caucasian_dec_s9 += 1
        if (dataset.race[i] == 'Other'):
            cnt_other_dec_s9 += 1
        if (dataset.race[i] == 'Native American'):
            cnt_native_dec_s9 += 1
        if (dataset.race[i] == 'Hispanic'):
            cnt_hispanic_dec_s9 += 1
        if (dataset.race[i] == 'Asian'):
            cnt_asian_dec_s9 += 1
    if (dataset.decile_score[i] == 10):
        cnt_dec_score10 += 1
        if (dataset.race[i] == 'African-American'):
            cnt_afr_am_dec_s10 += 1
        if (dataset.race[i] == 'Caucasian'):
            cnt_caucasian_dec_s10 += 1
        if (dataset.race[i] == 'Other'):
            cnt_other_dec_s10 += 1
        if (dataset.race[i] == 'Native American'):
            cnt_native_dec_s10 += 1
        if (dataset.race[i] == 'Hispanic'):
            cnt_hispanic_dec_s10 += 1
        if (dataset.race[i] == 'Asian'):
            cnt_asian_dec_s10 += 1

    if (dataset.sex[i] == 'Female'):
        if (dataset.decile_score[i] == 1):
            cnt_female_dec_s1 += 1
        if (dataset.decile_score[i] == 2):
            cnt_female_dec_s2 += 1
        if (dataset.decile_score[i] == 3):
            cnt_female_dec_s3 += 1
        if (dataset.decile_score[i] == 4):
            cnt_female_dec_s4 += 1
        if (dataset.decile_score[i] == 5):
            cnt_female_dec_s5 += 1
        if (dataset.decile_score[i] == 6):
            cnt_female_dec_s6 += 1
        if (dataset.decile_score[i] == 7):
            cnt_female_dec_s7 += 1
        if (dataset.decile_score[i] == 8):
            cnt_female_dec_s8 += 1
        if (dataset.decile_score[i] == 9):
            cnt_female_dec_s9 += 1
        if (dataset.decile_score[i] == 10):
            cnt_female_dec_s10 += 1

    if (dataset.sex[i] == 'Male'):
        if (dataset.decile_score[i] == 1):
            cnt_male_dec_s1 += 1
        if (dataset.decile_score[i] == 2):
            cnt_male_dec_s2 += 1
        if (dataset.decile_score[i] == 3):
            cnt_male_dec_s3 += 1
        if (dataset.decile_score[i] == 4):
            cnt_male_dec_s4 += 1
        if (dataset.decile_score[i] == 5):
            cnt_male_dec_s5 += 1
        if (dataset.decile_score[i] == 6):
            cnt_male_dec_s6 += 1
        if (dataset.decile_score[i] == 7):
            cnt_male_dec_s7 += 1
        if (dataset.decile_score[i] == 8):
            cnt_male_dec_s8 += 1
        if (dataset.decile_score[i] == 9):
            cnt_male_dec_s9 += 1
        if (dataset.decile_score[i] == 10):
            cnt_male_dec_s10 += 1

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
print("\n all races = ", all_races)

#distribution of decile scores on black and white
distributionRace = {'dec_score': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'African-American': [cnt_afr_am_dec_s1, cnt_afr_am_dec_s2, cnt_afr_am_dec_s3, cnt_afr_am_dec_s4, cnt_afr_am_dec_s5,
                      cnt_afr_am_dec_s6, cnt_afr_am_dec_s7, cnt_afr_am_dec_s8, cnt_afr_am_dec_s9, cnt_afr_am_dec_s10],
            'Caucasian': [cnt_caucasian_dec_s1, cnt_caucasian_dec_s2, cnt_caucasian_dec_s3, cnt_caucasian_dec_s4, cnt_caucasian_dec_s5,
                      cnt_caucasian_dec_s6, cnt_caucasian_dec_s7, cnt_caucasian_dec_s8, cnt_caucasian_dec_s9, cnt_caucasian_dec_s10],
            'Hispanic': [cnt_hispanic_dec_s1, cnt_hispanic_dec_s2, cnt_hispanic_dec_s3, cnt_hispanic_dec_s4, cnt_hispanic_dec_s5,
                         cnt_hispanic_dec_s6, cnt_hispanic_dec_s7, cnt_hispanic_dec_s8, cnt_hispanic_dec_s9, cnt_hispanic_dec_s10],
            'Asian': [cnt_asian_dec_s1, cnt_asian_dec_s2, cnt_asian_dec_s3, cnt_asian_dec_s4, cnt_asian_dec_s5,
                      cnt_asian_dec_s6, cnt_asian_dec_s7, cnt_asian_dec_s8, cnt_asian_dec_s9, cnt_asian_dec_s10],
            'Native': [cnt_native_dec_s1, cnt_native_dec_s2, cnt_native_dec_s3, cnt_native_dec_s4, cnt_native_dec_s5,
                       cnt_native_dec_s6, cnt_native_dec_s7, cnt_native_dec_s8, cnt_native_dec_s9, cnt_native_dec_s10],
            'Other': [cnt_other_dec_s1, cnt_other_dec_s2, cnt_other_dec_s3, cnt_other_dec_s4, cnt_other_dec_s5,
                      cnt_other_dec_s6, cnt_other_dec_s7, cnt_other_dec_s8, cnt_other_dec_s9, cnt_other_dec_s10]
                    }
distribution_dec_scores_race = pd.DataFrame(distributionRace, columns=['dec_score', 'African-American', 'Caucasian', 'Hispanic', 'Asian', 'Native', 'Other'])
print("\n distribution of decilescores on different races:\n", distribution_dec_scores_race)
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_afr = [cnt_afr_am_dec_s1, cnt_afr_am_dec_s2, cnt_afr_am_dec_s3, cnt_afr_am_dec_s4, cnt_afr_am_dec_s5, cnt_afr_am_dec_s6, cnt_afr_am_dec_s7, cnt_afr_am_dec_s8, cnt_afr_am_dec_s9, cnt_afr_am_dec_s10]
y_cau = [cnt_caucasian_dec_s1, cnt_caucasian_dec_s2, cnt_caucasian_dec_s3, cnt_caucasian_dec_s4, cnt_caucasian_dec_s5, cnt_caucasian_dec_s6, cnt_caucasian_dec_s7, cnt_caucasian_dec_s8, cnt_caucasian_dec_s9, cnt_caucasian_dec_s10]
y_his = [cnt_hispanic_dec_s1, cnt_hispanic_dec_s2, cnt_hispanic_dec_s3, cnt_hispanic_dec_s4, cnt_hispanic_dec_s5, cnt_hispanic_dec_s6, cnt_hispanic_dec_s7, cnt_hispanic_dec_s8, cnt_hispanic_dec_s9, cnt_hispanic_dec_s10]
y_asi = [cnt_asian_dec_s1, cnt_asian_dec_s2, cnt_asian_dec_s3, cnt_asian_dec_s4, cnt_asian_dec_s5, cnt_asian_dec_s6, cnt_asian_dec_s7, cnt_asian_dec_s8, cnt_asian_dec_s9, cnt_asian_dec_s10]
y_nat = [cnt_native_dec_s1, cnt_native_dec_s2, cnt_native_dec_s3, cnt_native_dec_s4, cnt_native_dec_s5, cnt_native_dec_s6, cnt_native_dec_s7, cnt_native_dec_s8, cnt_native_dec_s9, cnt_native_dec_s10]
y_oth = [cnt_other_dec_s1, cnt_other_dec_s2, cnt_other_dec_s3, cnt_other_dec_s4, cnt_other_dec_s5, cnt_other_dec_s6, cnt_other_dec_s7, cnt_other_dec_s8, cnt_other_dec_s9, cnt_other_dec_s10]



plt.bar(x,y_afr,align='center') # A bar chart
plt.xlabel('decile scores')
plt.ylabel('number of convicts')
plt.title('African-American distribution of decile scores')
plt.show()

plt.bar(x,y_cau,align='center') # A bar chart
plt.xlabel('decile scores')
plt.ylabel('number of convicts')
plt.title('Caucasian distribution of decile scores')
plt.show()

plt.bar(x,y_oth,align='center') # A bar chart
plt.xlabel('decile scores')
plt.ylabel('number of convicts')
plt.title('Other distribution of decile scores')
plt.show()

plt.bar(x,y_his,align='center') # A bar chart
plt.xlabel('decile scores')
plt.ylabel('number of convicts')
plt.title('Hispanic distribution of decile scores')
plt.show()

plt.bar(x,y_asi,align='center') # A bar chart
plt.xlabel('decile scores')
plt.ylabel('number of convicts')
plt.title('Asian distribution of decile scores')
plt.show()

plt.bar(x,y_nat,align='center') # A bar chart
plt.xlabel('decile scores')
plt.ylabel('number of convicts')
plt.title('Native American distribution of decile scores')
plt.show()




distributionSex = {'dec_score': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Female': [cnt_female_dec_s1, cnt_female_dec_s2, cnt_female_dec_s3, cnt_female_dec_s4, cnt_female_dec_s5,
                       cnt_female_dec_s6, cnt_female_dec_s7, cnt_female_dec_s8, cnt_female_dec_s9, cnt_female_dec_s10],
            'Male': [cnt_male_dec_s1, cnt_male_dec_s2, cnt_male_dec_s3, cnt_male_dec_s4, cnt_male_dec_s5,
                     cnt_male_dec_s6, cnt_male_dec_s7, cnt_male_dec_s8, cnt_male_dec_s9, cnt_male_dec_s10]
                   }

y_female = [cnt_female_dec_s1, cnt_female_dec_s2, cnt_female_dec_s3, cnt_female_dec_s4, cnt_female_dec_s5, cnt_female_dec_s6, cnt_female_dec_s7, cnt_female_dec_s8, cnt_female_dec_s9, cnt_female_dec_s10]
y_male = [cnt_male_dec_s1, cnt_male_dec_s2, cnt_male_dec_s3, cnt_male_dec_s4, cnt_male_dec_s5, cnt_male_dec_s6, cnt_male_dec_s7, cnt_male_dec_s8, cnt_male_dec_s9, cnt_male_dec_s10]

plt.bar(x,y_female,align='center') # A bar chart
plt.xlabel('decile scores')
plt.ylabel('number of convicts')
plt.title('Female distribution of decile scores')
plt.show()

plt.bar(x,y_male,align='center') # A bar chart
plt.xlabel('decile scores')
plt.ylabel('number of convicts')
plt.title('Male distribution of decile scores')
plt.show()

distribution_dec_scores_sex = pd.DataFrame(distributionSex, columns=['dec_score', 'Female', 'Male'])
print("\n distribution of decilescores on male/female:\n", distribution_dec_scores_sex)


