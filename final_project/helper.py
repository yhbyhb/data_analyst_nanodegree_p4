#!/usr/bin/python
'''
    helper functions for identifying person of interest
'''

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import *

def get_k_best_features(data_dict, features_list, k):
    '''
    Using SelectKBest, find k best features.
    returns list of features and list of scores
    '''
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)

    unsorted_pair_list = zip(features_list[1:], k_best.scores_)
    # print unsorted_dict_list
    sorted_pair_list = sorted(unsorted_pair_list, key=lambda x: x[1], reverse=True)
    # print sorted_dict_list
    k_best_features = [pair[0] for pair in sorted_pair_list]
    k_best_scores = [pair[1] for pair in sorted_pair_list]

    return k_best_features[:k], k_best_scores[:k]

def remove_outliers(data_dict, outliers):
    '''
    remove a list of outliers from data_dict
    '''
    for outlier in outliers:
        data_dict.pop(outlier, None)

def add_custum_features(data_dict, features_list):
    '''
    Add custom features to data_dict.

    total_income : salary + bonus + exercised_stock_options + total_stock_value
    ratio_poi_email : (from_poi_to_this_person + from_this_person_to_poi) / (to_messages + from_messages)
    has_email : boolean type. whether record has email account.
    '''
    mail_features = ['from_poi_to_this_person', 'from_this_person_to_poi',
                     'to_messages', 'from_messages']
    total_income_features = ['salary', 'bonus',
                             'exercised_stock_options', 'total_stock_value']

    for key in data_dict:
        has_nan = False
        record = data_dict[key]

        total_income = 0
        for feature in total_income_features:
            if record[feature] != 'NaN':
                total_income += record[feature]

        record['total_income'] = total_income

        for feature in mail_features:
            if record[feature] == 'NaN':
                has_nan = True
        if has_nan == False:
            record['ratio_poi_email'] = \
            (record['from_poi_to_this_person'] + record['from_this_person_to_poi']) / \
            float((record['to_messages'] + record['from_messages']))
        else:
            record['ratio_poi_email'] = 'NaN'

        # if record['email_address'] =='NaN':
        #     record['has_email'] = False
        # else:
        #     record['has_email'] = True
        # print record

    features_list += ['total_income', 'ratio_poi_email']

def find_best_parameters(pipeline, parameters, score_func, dataset, 
                         feature_list, test_size=0.2, n_iter=10):
    """
    find best parameter by using GridSearchCV with given scoring function.

    returns GridSearchCV object that has best parameters.
    """

    data = featureFormat(dataset, feature_list)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, 1, test_size=test_size, random_state = 42)
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

    sss = StratifiedShuffleSplit(labels_train, n_iter=n_iter , test_size=test_size, random_state=42)

    clf = GridSearchCV(pipeline, parameters, scoring=score_func, cv=sss, n_jobs=-1)
    clf.fit(features_train, labels_train)

    return clf

def validation(clf, dataset, feature_list, test_size=0.2, n_iter=1000):
    '''
    validate given classifier with using stratifie shuffle split cross validation. 
    returns average precision and recall
    '''
    data = featureFormat(dataset, feature_list)
    labels, features = targetFeatureSplit(data)

    precision = []
    recall = []

    cv = StratifiedShuffleSplit(labels, n_iter, test_size=test_size, random_state = 42)
    for train_idx, test_idx in cv:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)

        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))

    return np.mean(precision), np.mean(recall)
