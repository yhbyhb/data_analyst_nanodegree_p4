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

def add_custum_features(data_dict, features_list):
    '''
    Add custom features to data_dict.

    ratio_poi_email : (from_poi_to_this_person + from_this_person_to_poi) / (to_messages + from_messages)
    has_email : boolean type. whether record has email account.
    '''
    features = ['from_poi_to_this_person', 'from_this_person_to_poi',
                'to_messages', 'from_messages']

    for key in data_dict:
        has_nan = False
        record = data_dict[key]
        for feature in features:
            if record[feature] == 'NaN':
                has_nan = True

        if has_nan == False:
            record['ratio_poi_email'] = \
            (record['from_poi_to_this_person'] + record['from_this_person_to_poi']) / \
            float((record['to_messages'] + record['from_messages']))
        else:
            record['ratio_poi_email'] = 'NaN'

        if record['email_address'] =='NaN':
            record['has_email'] = False
        else:
            record['has_email'] = True
        # print record

    features_list += ['ratio_poi_email', 'has_email']