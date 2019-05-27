# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 08:46:11 2018

@author: Jerry
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.cross_validation import train_test_split


import os
os.chdir("C:\\Users\\Jerry\\Desktop\\university of waterloo\\3B\\MSCI 446 - Data Mining\\Project\\Final files")

#importing the data
mydata = pd.read_csv("Colour sheet with cluster and cut.csv")
column_names = list(mydata.columns)

data = mydata[column_names[2:]]
target = mydata[column_names[1]]

count_action = 0
count_comedy = 0
count_drama = 0
count_family = 0
count_horror = 0
for i in target:
    if i == "Action":
        count_action += 1
    elif i == "Comedy":
        count_comedy += 1
    elif i == "Drama":
        count_drama += 1
    elif i == "Family":
        count_family += 1
    elif i == "Horror":
        count_horror += 1



#using multinomial logistic regression
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')


from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

stratification = StratifiedKFold(y = target, n_folds = 10, shuffle = True, random_state=1)
scores = cross_val_score(mul_lr, data, target, cv = stratification)
scores.mean()

target_predict = cross_val_predict(mul_lr, data, target, cv = stratification)
confusion_matrix(target, target_predict)


#splitting into training and testing data
#train_x, test_x, train_y, test_y = train_test_split(data,target, train_size = 0.7)

#mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_x, train_y)
#print ("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))
#print ("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))

#lr = linear_model.LogisticRegression().fit(train_x, train_y)
#print ("Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, lr.predict(train_x)))
#print ("Logistic regression Test Accuracy :: ", metrics.accuracy_score(test_y, lr.predict(test_x)))



#######testing the genres individually

#family
family = list()
for i in target:
    if i == 'Family':
        family.append(1)
    else:
        family.append(0)

stratification = StratifiedKFold(y = family, n_folds = 10, shuffle = True, random_state=1)
lr = linear_model.LogisticRegression()
scores_family = cross_val_score(lr, data, family, cv = stratification)
scores_family.mean()

family_predict = cross_val_predict(lr, data, family, cv = stratification)
confusion_matrix(family, family_predict)


#comedy
comedy = list()
for i in target:
    if i == 'Comedy':
        comedy.append(1)
    else:
        comedy.append(0)

stratification = StratifiedKFold(y = comedy, n_folds = 10, shuffle = True, random_state=1)
lr = linear_model.LogisticRegression()
scores_comedy = cross_val_score(lr, data, comedy, cv = stratification)
scores_comedy.mean()

comedy_predict = cross_val_predict(lr, data, comedy, cv = stratification)
confusion_matrix(comedy, comedy_predict)

#action
action = list()
for i in target:
    if i == 'Action':
        action.append(1)
    else:
        action.append(0)

stratification = StratifiedKFold(y = action, n_folds = 10, shuffle = True, random_state=1)
lr = linear_model.LogisticRegression()
scores_action = cross_val_score(lr, data, action, cv = stratification)
scores_action.mean()

action_predict = cross_val_predict(lr, data, action, cv = stratification)
confusion_matrix(action, action_predict)

#drama
drama = list()
for i in target:
    if i == 'Drama':
        drama.append(1)
    else:
        drama.append(0)

stratification = StratifiedKFold(y = drama, n_folds = 10, shuffle = True, random_state=1)
lr = linear_model.LogisticRegression()
scores_drama = cross_val_score(lr, data, drama, cv = stratification)
scores_drama.mean()

drama_predict = cross_val_predict(lr, data, drama, cv = stratification)
confusion_matrix(drama, drama_predict)

#horror
horror = list()
for i in target:
    if i == 'Horror':
        horror.append(1)
    else:
        horror.append(0)

stratification = StratifiedKFold(y = horror, n_folds = 10, shuffle = True, random_state=1)
lr = linear_model.LogisticRegression()
scores_horror = cross_val_score(lr, data, horror, cv = stratification)
scores_horror.mean()

horror_predict = cross_val_predict(lr, data, horror, cv = stratification)
confusion_matrix(horror, horror_predict)

#groups of action, comedy, other
modified = list()
for i in target:
    if i == 'Drama' or i == 'Family' or i == 'Horror':
        modified.append('Other')
    elif i == 'Action':
        modified.append('Action')
    elif i == 'Comedy':
        modified.append('Comedy')

stratification = StratifiedKFold(y = modified, n_folds = 10, shuffle = True, random_state=1)
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
scores_modified = cross_val_score(mul_lr, data, modified, cv = stratification)
scores_modified.mean()
   
modified_predict = cross_val_predict(lr, data, modified, cv = stratification)
confusion_matrix(modified, modified_predict)



