#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

from sklearn.feature_selection import SelectKBest

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

